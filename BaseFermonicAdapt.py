from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.drivers import PySCFDriver
from typing import List
from qiskit import Aer
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD,UCC
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator  
import numpy as np
import logging
import sys,pickle,copy,re
sys.path.append('..')
from qiskit_nature.second_q.drivers import PySCFDriver

class BaseFermonicAdaptVQE():
    """
    咱就是说 这个类是用来做什么的 是个基类
    
    """
    #
    def __init__(self, ES_problem: ElectronicStructureProblem,threshold:float=1e-3,max_iter:int=40) -> None:
        self.mapper = JordanWignerMapper()
        self.es_problem = ES_problem
        self.estimator = Estimator()
        
        self.threshold = threshold
        self.max_iteration = max_iter
        self.hamiltonian = self.mapper.map(ES_problem.hamiltonian.second_q_op())
        self.init_state_hf = HartreeFock(num_particles=self.es_problem.num_particles,
                                         num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                                         qubit_mapper=self.mapper)
        self.n_qubit = self.init_state_hf.num_qubits
        self.adapt_ansatz=QuantumCircuit(self.n_qubit)
        self.adapt_ansatz.append(self.init_state_hf,range(self.n_qubit))#存放动态增长的ansatz circuit的instruction
        self.converageflag = False #是否收敛的标志
        self._already_pick_index = [] #目前已经挑选的index列表
        self.finnal_pool_op=[] 
        self.solver = VQE(estimator=Estimator(),
                          ansatz=self.init_state_hf,
                          optimizer=SLSQP())
        self.fermonic_pool_init()
        self.first_step()

    def fermonic_pool_init(self):
        """
        这个函数是用来初始化fermonic算符池的 
        由于是FermonicAdapt 所以UCC里找的是sd激发 
        
        数理基础其实重要 比如回答问题:
        1.为什么用UCC的sd激发的时候要指定initial_state?不然会怎样？
        2.为什么要commutor要乘以i来保证结果是实数? e^iA 为什么有i?
        
        编程问题:
        1.列表解析式的使用？
        2.如果不使用parameter_prefix='{:02d}'.format(index)会有什么后果(重要！)
        3.如何实现热启动 (这里没有实现 你试着实现 )
        """
        
        uccsd = UCC(num_particles=self.es_problem.num_particles,
                    num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                    excitations='sd',qubit_mapper=JordanWignerMapper(),initial_state=self.init_state_hf)
        
        self.fermonic_pool = [EvolvedOperatorAnsatz(operators=JordanWignerMapper().map(i),name='Fermonic_op'+'{:02d}'.format(index),\
            parameter_prefix='{:02d}'.format(index)) for index,i in enumerate(uccsd.excitation_ops())]
        self.finnal_pool_op = [self.mapper.map(i) for i in uccsd.excitation_ops()]
        print(f'fermonicpool 创建完毕,size={len(self.finnal_pool_op)}个')
        self.commutors= [1j * (self.hamiltonian @ exc - exc @ self.hamiltonian) for exc in self.finnal_pool_op]
        
        
        
        
    @staticmethod
    # 参数是 每一轮求得的梯度的最大值
    #判断是否达到收敛条件
    def check_gradient_converge(value:List,criterion: float = 1e-3) -> bool:
        converge = value.max()
        if converge > criterion:
            print(f'没有达到收敛标准,标准为{criterion},当前值为{converge}')
            return False
        else:
            print(f'达到收敛标准,标准为{criterion},当前值为{converge}')
            return True
            
    def first_step(self):
        """
        这个函数用来进行第一步的初始化 选取出第一个算符 由于基于HF 
        因此是特殊的 最好单独写 反正我这猪脑子想不到咋把第一步也一步到位写好
        
        """
        print('现在挑选第一个算符...')
        circuits = []
        n = self.n_qubit
        for i in range(len(self.finnal_pool_op)): 
            qc = QuantumCircuit(n)
            qc.append(self.init_state_hf, range(n))
            circuits.append(qc)
        job = self.estimator.run(circuits=circuits,observables=self.commutors)
        
        result = job.result()
        value =np.abs(result.values)
        
        k = np.argmax(value)
        print(f'初始化结果:第{np.argmax(value)}项被选定,此项梯度最大,为{value[k]}')
        self._already_pick_index.append(k)
        self.iteration_index = int(1)
        self.next_operator= EvolvedOperatorAnsatz(operators=self.finnal_pool_op[k],parameter_prefix='{:02d}'.format(0),name="Fermonic"+'_'+str(k))

        #此时更新adapt_ansatz
        self.adapt_ansatz.append(self.next_operator,range(self.n_qubit))


        self.solver.ansatz = self.adapt_ansatz
        self.solver.initial_point=[0.0]
        self.vqe_result = self.solver.compute_minimum_eigenvalue(self.hamiltonian)
        self.optimal_parameter = self.vqe_result.optimal_point.tolist()
        print(f'第一轮的优化结果:optimal_point={self.vqe_result.optimal_point}')
    
    
    def next_operator(self, bound_circuit: QuantumCircuit):
        print(f'No.{self.iteration_index}轮,正在准备挑选下一块算符...')
        job = self.estimator.run(circuits=[bound_circuit]*len(self.commutors), observables=self.commutors)
        result = job.result()
        value = np.abs(result.values)
        k = np.argmax(value)
        #判断即将选择的算符是否已经陷入循环
        if self.check_gradient_converge(value=value,criterion=self.threshold):
            print(f'梯度满足或者检测到循环!')
            self.converageflag=True
            print(f'已经达到收敛条件!')
            return

        else:
            print(f'第{self.iteration_index}轮中梯度最大项为第{k}项,已被选入算符池...')  
            self.optimal_parameter.append(0.0)      
            self.adapt_ansatz.append(EvolvedOperatorAnsatz(operators=self.finnal_pool_op[k],
                                                           parameter_prefix='{:02d}'.format(self.iteration_index),name='Fermonic_'+str(k)),range(self.n_qubit))
            self._already_pick_index.append(k)

        

    
    def run(self):
        #vqe_result=[]
        while(self.converageflag==False and self.iteration_index<self.max_iteration):
            print(f'------------第{self.iteration_index}轮--------------')
            print(f'已经选好的index是{self._already_pick_index}')

            self.solver.ansatz = self.adapt_ansatz
            self.solver.initial_point=self.optimal_parameter            
            self.next_operator(bound_circuit=self.solver.ansatz.bind_parameters(self.optimal_parameter))
            self.vqe_result = self.solver.compute_minimum_eigenvalue(self.hamiltonian)
            self.iteration_index += 1
        
        self.converageflag=True
        print(f'✔Adapt VQE算法结果={self.vqe_result.optimal_value}')

        

                
            
            
            
    