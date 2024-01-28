from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.drivers import PySCFDriver
from typing import List,Sequence
from qiskit.circuit import Instruction, InstructionSet
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD,UCC
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP,COBYLA
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.opflow import StateFn, CircuitStateFn,I,X,Y,Z
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from copy import deepcopy
from qiskit.quantum_info import Pauli
from qiskit.circuit import ParameterVector, Parameter
from qiskit.primitives import Estimator  # 意味着本地进行的模拟
import numpy as np
from qiskit.algorithms.optimizers import SPSA
import logging
import time
import copy


class QubitAdapt(object):
    #qubit-adapt 从UCCSD 以及UCCGSD(未实现)的Pauli string中提取算符作为qubit pool
    def __init__(self, ES_problem: ElectronicStructureProblem,Completminimum_pool_isused:bool = None) -> None:
        self.logger_init() #初始化日志
        self.converter = QubitConverter(JordanWignerMapper())
        self.es_problem = ES_problem
        self.init_state_hf = HartreeFock(num_particles=self.es_problem.num_particles,
                                         num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                                         qubit_converter=self.converter)
        self.n_qubit = self.init_state_hf.num_qubits
        self.hamiltonian = self.converter.convert(ES_problem.hamiltonian.second_q_op())
        uccsd = UCCSD(qubit_converter=self.converter,
                      num_particles=self.es_problem.num_particles,
                      num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                      initial_state=self.init_state_hf,
                      generalized=False  # 如果设为True ==> UCCGSD G=generalized
                      )
        self.uccop = [self.converter.convert(i) for i in uccsd.excitation_ops()]
        self.uccop_dict = [self.pauliOperator2Dict(pauliOperator=i) for i in self.uccop]
        self.qubit_pool_init()
        self._getCompleteMinimalPool()
        if Completminimum_pool_isused==True:
            #把算符池换成CMpool
            self.finnal_pool_instruction = self.cmpool_instruction
            self.commutors = [1j*(self.hamiltonian@i - i@self.hamiltonian) for i in self.cmpool]
            self.logger.info('你最终选择了最小集算符池作为算符最终算符池!')
        else:
            self.finnal_pool_instruction = self.paulistring_withoutz_instruction
            self.commutors = [1j*(self.hamiltonian@i - i@self.hamiltonian) for i in self.qubit_pool_withoutz]
            self.logger.info('你最终选择了qubit_string算符池作为算符最终算符池!')
        self.adapt_ansatz=[]
        self.converageflag = False
        
        
        self._already_pick_index = []
        self.slover = VQE(estimator=Estimator(),ansatz=self.init_state_hf,optimizer=SLSQP())
        self.adapt_ansatz.append(self.circuit_measurement_first())

        
    @staticmethod
    def pauliOperator2Dict(pauliOperator):
        paulis = pauliOperator.to_pauli_op().oplist
        paulis_dict = {}
        for x in paulis:
            if x.coeff == 0.0:
                continue
            label = x.primitive.to_label()
            coeff = x.coeff
            paulis_dict[label] = coeff

        return paulis_dict
    @staticmethod
    def random_pick(pre_decied_array:List,propotion:float=0.5):
        #example:propotion = 0.25
        if propotion>1:
            raise ValueError('propotion必须小于等于1!')
        
        pick_num = int(len(pre_decied_array)*propotion)
        np.random.seed(42)
        pick_index = np.random.randint(0,len(pre_decied_array),pick_num)
        return [pre_decied_array[i] for i in pick_index]
    
        
    
    def qubit_pool_init(self):
        tmp = [list(i) for i in self.uccop_dict]
        self.paulistring_withz=[]
        for i in tmp:
            if len(i)==2:
                i.pop()
            elif len(i)==8:
                del i[4:8]
            self.paulistring_withz.append(i)
        self.propotion = 0.25
        self.paulistring_withoutz = [k.replace('Z','I')  for i in self.paulistring_withz for k in i]
        if len(self.paulistring_withoutz)>10:
            self.logger.info(f'由于算符池较大,进行随机分割! ')  
            self.paulistring_withoutz_random = self.random_pick(self.paulistring_withoutz,propotion=self.propotion)
        else:
            self.logger.info(f'由于算符池过小,不进行随机分割!') 
            self.paulistring_withoutz_random = self.paulistring_withoutz
        #qubit_pool_withoutz 用于构建comuutors
        self.qubit_pool_withoutz = [PauliOp(primitive=Pauli(data=i),coeff=1j) for i in self.paulistring_withoutz_random]
        self.paulistring_withoutz_instruction = [EvolvedOperatorAnsatz(operators=PauliOp(Pauli(data=i),coeff=(1.0)),parameter_prefix="term_"+str(index),name='qubit_string_'+str(index)).to_instruction() \
                                                 for index,i in enumerate(self.paulistring_withoutz_random)]
    
        self.logger.info(f'初始化了Paulistring池,完整池{len(self.paulistring_withoutz)},Propotion=0.25,实际池{len(self.paulistring_withoutz_instruction)}')
    
    def _getCompleteMinimalPool_1(self, n):
        if n < 3:
            raise ValueError("Minimal pool has at least 3 qubits")
        if n == 3:
            return [Z^Z^Y, Z^Y^I, Y^I^I, I^Y^I]
        
        oldPool = self._getCompleteMinimalPool_1(n-1)
        nY = 1
        nYminus1 = 1
        for i in range(n-2):
            nY = nY^I
            nYminus1 = nYminus1^I
        nY = (Y^ nY ^ I )
        nYminus1 = (I^Y^nYminus1)
        newPool = [Z^x for x in oldPool]
        newPool.append(nY)
        newPool.append(nYminus1)
        return newPool
    
    def _getCompleteMinimalPool(self):
        self.cmpool =  self._getCompleteMinimalPool_1(n=self.n_qubit)
        self.cmpool_instruction = [EvolvedOperatorAnsatz(operators=PauliOp(Pauli(data=str(i)),coeff=(1.0)),parameter_prefix="term_"+str(index),name='cmpool_'+str(index)).to_instruction() \
                                                  for index,i in enumerate(self.cmpool)]
        self.cmpool_circuit = [EvolvedOperatorAnsatz(operators=PauliOp(Pauli(data=str(i)),coeff=(1.0)),parameter_prefix="term_"+str(index),name='cmpool_'+str(index)) \
                                                  for index,i in enumerate(self.cmpool)]
        self.logger.info(f'初始化了最小算符池!,共计{len(self.cmpool_instruction)}项算符池成员!')
        
    @staticmethod
    # 参数是 每一轮求得的梯度的最大值
    def check_gradient_converge(value, logger,criterion: float = 1e-3) -> bool:
        converge = value[np.argmax(value)]
        if converge > criterion:
            logger.info(f'没有达到收敛标准,标准为{criterion},当前值为{converge}')
            return False
        else:
            logger.info(f'达到收敛标准,标准为{criterion},当前值为{converge}')
            return True
    
    #挑选下一块
    def pick_next_operator(self, bound_circuit: QuantumCircuit):
        self.logger.info(f'目前这是第{self.iteration_index}轮,正在准备挑选下一块算符,并验证是否收敛...')
        estimator = Estimator()
        job = estimator.run(circuits=[bound_circuit]*len(self.commutors), observables=self.commutors)
        result = job.result()
        value = np.abs(result.values)
        #self.logger.info(f'第{self.iteration_index}轮的结果:{value}')
        self.converageflag = self.check_gradient_converge(value=value,logger=self.logger)
        if self.converageflag == False:
            self.logger.info(f'目前还没有达到收敛！')
            k = np.argmax(value)
            self.logger.info(f'第{self.iteration_index}轮中梯度最大项为第{k}项,已被选入算符池...')
            
            #print(f'第{self.iteration_index}轮中梯度最大项为第{k}项,梯度最小项为第{np.argmin(value)}项')
            if k in self._already_pick_index: 
                self.logger.info(f'出现了重复的元素!尝试改名!')
                tmp = copy.deepcopy(self.finnal_pool_instruction)
                tmp[k]._params[0]._name='picked_'+str(self.iteration_index)+'_'+str(k)+'_[0]'
                self.adapt_ansatz.append(tmp[k])
                print(f'加入的是:{tmp[k]}')
            else:
                self.adapt_ansatz.append(self.finnal_pool_instruction[k])
            self._already_pick_index.append(k)
            return result.values
        else:
            self.logger.info(f'已经达到收敛！算法终止！')
            # print(f'第{self.iteration_index}轮已经收敛！')
    
    def logger_init(self):
        # 定义记录器对象
        self.logger = logging.getLogger('Qubit_AdaptVQE')
        # 设置记录器级别
        self.logger.setLevel(logging.DEBUG)
        # 设置过滤器 只有被选中的可以记录
        myfilter = logging.Filter('Qubit_AdaptVQE')
        # 定义处理器-文件处理器
        filehandler = logging.FileHandler(filename='./Qubit_AdaptVQE.log', mode='w')
        filehandler.addFilter(myfilter)
        formatter = logging.Formatter('%(asctime)s-%(levelname)s-\n%(message)s')
        filehandler.setFormatter(formatter)
        # 定义处理器-控制台处理器
        concolehander = logging.StreamHandler()
        concolehander.setLevel(logging.INFO)
        # 记录器绑定handerler
        self.logger.addHandler(filehandler)
        self.logger.addHandler(concolehander)
        self.logger.info('logger init done!')
        
    
    def circuit_measurement_first(self):
        self.logger.info('开始初始化:挑选第一个算符...')
        circuits = []
        for i in self.finnal_pool_instruction:
            n = self.init_state_hf.num_qubits
            qc = QuantumCircuit(n)
            init_state = deepcopy(self.init_state_hf)
            qc.append(init_state, range(n))
            circuits.append(qc)

        estimator = Estimator()
        job = estimator.run(circuits=circuits, observables=self.commutors)
        result = job.result()
        value = np.abs(result.values)
        k = np.argmax(value)
        #self.logger.info(f'初始化结果:\n{value}')
        self.logger.info(f'初始化结果:第{np.argmax(value)}项被选定,此项梯度最大,为{value[k]}')
        self._already_pick_index.append(k)
        # print(f'初始化结果:第{np.argmax(value)}项被选定，梯度最大,梯度最小的是第{np.argmin(value)}项')
        self.iteration_index = 1
        tmp = copy.deepcopy(self.finnal_pool_instruction[k])
        # tmp._params[0]._name='picked_1_'+str(k)+'_[0]'
        return tmp
    
    
    def parameter_optimizer(self, parameterized_circuit: QuantumCircuit, maxiteration: int = 300, initial_parameter=None):
        if initial_parameter == None:
            initial_parameter = np.zeros(parameterized_circuit.num_parameters)
        # 输入为参数化线路 但尚未绑定参数：

        def _loss(x):
            bound_circuit = parameterized_circuit.bind_parameters(x)
            estimator = Estimator()
            job = estimator.run(circuits=bound_circuit,
                                observables=self.hamiltonian)
            result = job.result()
            return result.values[0]
        # print(f'初始设定参数值为{initial_parameter}')
        self.logger.info(f'初始设定参数值为{initial_parameter} 即将开始参数优化...')
        spsa = SPSA(maxiter=maxiteration)
        optimize_result = spsa.minimize(fun=_loss, x0=np.zeros(parameterized_circuit.num_parameters))
        self.logger.info(
            f'第{parameterized_circuit.num_parameters}轮的参数优化结果为{optimize_result.x}')
        # print(f'第{parameterized_circuit.num_parameters+1}轮的参数优化结果为{optimize_result.x}')
        return optimize_result.x
    
        
    def run_slover(self):
        optimal_parameter= [0.0]
        #vqe_result=[]
        while(self.converageflag==False):
            self.slover.ansatz =None
            start_time =time.time()
            self.logger.info(f'------------第{self.iteration_index}轮正在进行中--------------')
            self.logger.info(f'**目前已有{self._already_pick_index}**')
            qc = QuantumCircuit(self.n_qubit)
            qc.append(self.init_state_hf, range(self.n_qubit))
            for index, k in enumerate(self.adapt_ansatz):
                qc.append(k, range(self.n_qubit))
                
            # ------------------------------------------------------------
            # 每一轮的优化
            self.slover.ansatz = qc
            #self.slover.initial_point = np.zeros(qc.num_parameters)
            if len(optimal_parameter)==1:
                self.slover.initial_point=[0.0]
            else:
                self.slover.initial_point=optimal_parameter
            self.logger.info(f'initial point是:{self.slover.initial_point}')
            self.vqe_result = self.slover.compute_minimum_eigenvalue(self.hamiltonian)
            optimal_parameter = self.vqe_result.optimal_point.tolist()
            self.logger.info(f'第{self.iteration_index}轮的优化结果是：{optimal_parameter}')
            bound_circuit = qc.bind_parameters(optimal_parameter)
            self.pick_next_operator(bound_circuit=bound_circuit)
            optimal_parameter.append(0.0)
            self.iteration_index += 1
        self.logger.info(
            f'===FINAL OUTCOME===\nOrder={self._already_pick_index}\nOptimal value={optimal_parameter}\nTotal iteration={self.iteration_index-1}')
        return self.vqe_result.optimal_value


    
        