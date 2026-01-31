"""
参数验证器模块

提供高级的参数约束检查和验证功能
"""

from typing import Dict, Any, List, Callable, Optional, Tuple
import re
import numpy as np
from autodl_core import ParameterSpace, ParameterConfig, ParameterType


class ConstraintViolationError(Exception):
    """参数约束违反异常"""
    pass


class ParameterValidator:
    """
    参数验证器
    
    提供复杂的参数约束检查和验证功能
    """
    
    def __init__(self, parameter_space: ParameterSpace):
        self.parameter_space = parameter_space
        self.constraint_functions: Dict[str, Callable] = {}
        self._setup_default_constraints()
    
    def _setup_default_constraints(self):
        """设置默认的约束函数"""
        
        # 模型结构约束：hidden层应该递减
        def hidden_layers_constraint(params: Dict[str, Any]) -> bool:
            """隐藏层维度应该递减"""
            if all(key in params for key in ['dimensions', 'hidden1', 'hidden2']):
                return params['dimensions'] >= params['hidden1'] >= params['hidden2']
            return True
        
        # 解码器约束：解码器维度应该大于等于编码器最后一层
        def decoder_constraint(params: Dict[str, Any]) -> bool:
            """解码器维度约束"""
            if all(key in params for key in ['hidden2', 'decoder1']):
                return params['decoder1'] >= params['hidden2']
            return True
        
        # 注意力头数约束：应该能整除对应的隐藏维度
        def attention_heads_constraint(params: Dict[str, Any]) -> bool:
            """注意力头数应该合理"""
            constraints_passed = True
            
            # GAT头数约束
            if all(key in params for key in ['hidden1', 'gat_heads']):
                if params['hidden1'] % params['gat_heads'] != 0:
                    constraints_passed = False
            
            # GT头数约束  
            if all(key in params for key in ['hidden2', 'gt_heads']):
                if params['hidden2'] % params['gt_heads'] != 0:
                    constraints_passed = False
            
            # 融合头数约束
            if all(key in params for key in ['hidden2', 'fusion_heads']):
                if params['hidden2'] % params['fusion_heads'] != 0:
                    constraints_passed = False
            
            return constraints_passed
        
        # 学习率和权重衰减的合理性约束
        def learning_constraint(params: Dict[str, Any]) -> bool:
            """学习率和权重衰减的合理性"""
            if all(key in params for key in ['lr', 'weight_decay']):
                # 权重衰减不应该比学习率大太多
                return params['weight_decay'] <= params['lr'] * 10
            return True
        
        # 损失权重约束：至少有一个权重大于0
        def loss_weights_constraint(params: Dict[str, Any]) -> bool:
            """损失权重约束"""
            if all(key in params for key in ['alpha', 'beta', 'gamma']):
                return params['alpha'] > 0 or params['beta'] > 0 or params['gamma'] > 0
            return True
        
        # 批大小和MoCo队列大小的关系约束
        def batch_moco_constraint(params: Dict[str, Any]) -> bool:
            """批大小和MoCo队列大小约束"""
            if all(key in params for key in ['batch', 'moco_K']):
                # MoCo队列应该比批大小大很多
                return params['moco_K'] >= params['batch'] * 4
            return True
        
        # MoCo DoubleTau温度约束：tau2应该大于等于tau1
        def moco_tau_constraint(params: Dict[str, Any]) -> bool:
            """DoubleTau MoCo温度约束：tau2应该大于等于tau1"""
            if all(key in params for key in ['moco_tau1', 'moco_tau2']):
                tau1 = float(params['moco_tau1'])
                tau2 = float(params['moco_tau2'])
                return tau2 >= tau1
            return True
        
        # MoCo动量系数范围约束
        def moco_momentum_range_constraint(params: Dict[str, Any]) -> bool:
            """MoCo动量系数应该在合理范围内（0.9-0.9999）"""
            if 'moco_momentum' in params:
                momentum = float(params['moco_momentum'])
                return 0.9 <= momentum <= 0.9999
            return True
        
        # MoCo温度参数正值约束
        def moco_temperature_positive_constraint(params: Dict[str, Any]) -> bool:
            """所有MoCo温度参数应该为正值"""
            temp_params = ['moco_t', 'moco_tau1', 'moco_tau2']
            for param in temp_params:
                if param in params:
                    temp = float(params[param])
                    if temp <= 0:
                        return False
            return True
        
        # 注册默认约束
        self.constraint_functions.update({
            'hidden_layers_decreasing': hidden_layers_constraint,
            'decoder_size': decoder_constraint,
            'attention_heads_divisible': attention_heads_constraint,
            'learning_rates_reasonable': learning_constraint,
            'loss_weights_positive': loss_weights_constraint,
            'batch_moco_ratio': batch_moco_constraint,
            'moco_tau_ordering': moco_tau_constraint,
            'moco_momentum_range': moco_momentum_range_constraint,
            'moco_temperature_positive': moco_temperature_positive_constraint
        })
    
    def add_constraint(self, name: str, constraint_func: Callable[[Dict[str, Any]], bool]):
        """添加自定义约束函数"""
        self.constraint_functions[name] = constraint_func
    
    def validate_parameters(self, parameters: Dict[str, Any], 
                          strict: bool = True) -> Tuple[bool, List[str]]:
        """
        验证参数组合
        
        Args:
            parameters: 参数字典
            strict: 是否严格模式（严格模式下任何约束违反都返回False）
            
        Returns:
            (is_valid, error_messages): 验证结果和错误信息列表
        """
        errors = []
        
        # 基本参数空间验证
        if not self.parameter_space.validate_parameters(parameters):
            errors.append("参数不在定义的参数空间范围内")
            if strict:
                return False, errors
        
        # 约束函数验证
        for constraint_name, constraint_func in self.constraint_functions.items():
            try:
                if not constraint_func(parameters):
                    errors.append(f"违反约束: {constraint_name}")
                    if strict:
                        return False, errors
            except Exception as e:
                errors.append(f"约束检查错误 {constraint_name}: {str(e)}")
                if strict:
                    return False, errors
        
        return len(errors) == 0, errors
    
    def suggest_parameter_fix(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        建议参数修复
        
        尝试自动修复违反约束的参数
        """
        fixed_params = parameters.copy()
        
        # 首先修复基本参数空间问题
        for param_name, value in fixed_params.items():
            if param_name in self.parameter_space.parameters:
                config = self.parameter_space.parameters[param_name]
                
                # 修复连续型参数
                if config.param_type == ParameterType.CONTINUOUS:
                    val = float(value)
                    if val < config.bounds[0]:
                        fixed_params[param_name] = config.bounds[0]
                    elif val > config.bounds[1]:
                        fixed_params[param_name] = config.bounds[1]
                    else:
                        fixed_params[param_name] = val
                
                # 修复离散型参数
                elif config.param_type == ParameterType.DISCRETE:
                    if value not in config.values:
                        # 选择最接近的有效值
                        if isinstance(value, (int, float)):
                            closest = min(config.values, key=lambda x: abs(x - value))
                            fixed_params[param_name] = closest
                        else:
                            fixed_params[param_name] = config.values[0]
                
                # 修复分类型参数
                elif config.param_type == ParameterType.CATEGORICAL:
                    if value not in config.values:
                        fixed_params[param_name] = config.values[0]
        
        # 修复隐藏层递减约束
        if all(key in fixed_params for key in ['dimensions', 'hidden1', 'hidden2']):
            dimensions = int(fixed_params['dimensions'])
            hidden1 = int(fixed_params['hidden1'])
            hidden2 = int(fixed_params['hidden2'])
            
            if not (dimensions >= hidden1 >= hidden2):
                # 重新调整隐藏层大小，保持递减关系
                fixed_params['hidden1'] = min(hidden1, dimensions)
                fixed_params['hidden2'] = min(hidden2, int(fixed_params['hidden1']))
        
        # 修复解码器约束
        if all(key in fixed_params for key in ['hidden2', 'decoder1']):
            hidden2 = int(fixed_params['hidden2'])
            decoder1 = int(fixed_params['decoder1'])
            if decoder1 < hidden2:
                fixed_params['decoder1'] = max(decoder1, hidden2)
        
        # 修复注意力头数约束
        for hidden_key, heads_key in [('hidden1', 'gat_heads'), ('hidden2', 'gt_heads'), ('hidden2', 'fusion_heads')]:
            if all(key in fixed_params for key in [hidden_key, heads_key]):
                hidden_dim = int(fixed_params[hidden_key])
                heads = int(fixed_params[heads_key])
                
                # 找到最接近的可整除的头数
                possible_heads = [h for h in [2, 4, 8, 16] if hidden_dim % h == 0]
                if possible_heads:
                    # 选择最接近原始值的头数
                    fixed_params[heads_key] = min(possible_heads, key=lambda x: abs(x - heads))
                else:
                    # 如果没有可整除的头数，调整隐藏层维度
                    # 找到最接近的可被常用头数整除的维度
                    for target_heads in [2, 4, 8]:
                        if hidden_dim >= target_heads:
                            adjusted_dim = (hidden_dim // target_heads) * target_heads
                            if adjusted_dim > 0:
                                fixed_params[hidden_key] = adjusted_dim
                                fixed_params[heads_key] = target_heads
                                break
        
        # 修复学习率约束
        if all(key in fixed_params for key in ['lr', 'weight_decay']):
            lr = float(fixed_params['lr'])
            weight_decay = float(fixed_params['weight_decay'])
            if weight_decay > lr * 10:
                fixed_params['weight_decay'] = lr * 5
        
        # 修复损失权重约束
        if all(key in fixed_params for key in ['alpha', 'beta', 'gamma']):
            alpha = float(fixed_params['alpha'])
            beta = float(fixed_params['beta'])
            gamma = float(fixed_params['gamma'])
            if alpha <= 0 and beta <= 0 and gamma <= 0:
                # 至少设置一个权重为正值
                fixed_params['alpha'] = 1.0
        
        # 修复批大小和MoCo队列约束
        if all(key in fixed_params for key in ['batch', 'moco_K']):
            batch = int(fixed_params['batch'])
            moco_K = int(fixed_params['moco_K'])
            if moco_K < batch * 4:
                fixed_params['moco_K'] = batch * 4
        
        # 修复MoCo DoubleTau温度约束（tau2 >= tau1）
        if all(key in fixed_params for key in ['moco_tau1', 'moco_tau2']):
            tau1 = float(fixed_params['moco_tau1'])
            tau2 = float(fixed_params['moco_tau2'])
            if tau2 < tau1:
                # 调整tau2使其等于tau1，保持最小的修改
                fixed_params['moco_tau2'] = tau1
        
        # 修复MoCo动量系数范围约束
        if 'moco_momentum' in fixed_params:
            momentum = float(fixed_params['moco_momentum'])
            if momentum < 0.9:
                fixed_params['moco_momentum'] = 0.9
            elif momentum > 0.9999:
                fixed_params['moco_momentum'] = 0.9999
        
        # 修复MoCo温度参数正值约束
        temp_params = ['moco_t', 'moco_tau1', 'moco_tau2']
        for param in temp_params:
            if param in fixed_params:
                temp = float(fixed_params[param])
                if temp <= 0:
                    # 设置为默认的合理温度值
                    if param == 'moco_t':
                        fixed_params[param] = 0.2
                    elif param == 'moco_tau1':
                        fixed_params[param] = 0.2
                    elif param == 'moco_tau2':
                        fixed_params[param] = 0.3
        
        return fixed_params
    
    def get_constraint_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取约束检查报告
        
        返回详细的约束检查结果
        """
        report = {
            'parameters': parameters,
            'overall_valid': True,
            'constraint_results': {},
            'errors': [],
            'warnings': []
        }
        
        # 检查每个约束
        for constraint_name, constraint_func in self.constraint_functions.items():
            try:
                result = constraint_func(parameters)
                report['constraint_results'][constraint_name] = {
                    'passed': result,
                    'description': constraint_func.__doc__ or constraint_name
                }
                
                if not result:
                    report['overall_valid'] = False
                    report['errors'].append(f"约束违反: {constraint_name}")
                    
            except Exception as e:
                report['constraint_results'][constraint_name] = {
                    'passed': False,
                    'error': str(e),
                    'description': constraint_func.__doc__ or constraint_name
                }
                report['overall_valid'] = False
                report['errors'].append(f"约束检查错误 {constraint_name}: {str(e)}")
        
        return report


class ConfigurationConverter:
    """
    配置转换器
    
    将优化参数转换为实验配置格式
    """
    
    def __init__(self, task_type: str = "LDA"):
        self.task_type = task_type
        self.base_config = self._get_base_config()
    
    def _get_base_config(self) -> Dict[str, Any]:
        """获取基础配置"""
        return {
            'seed': 42,
            'epochs': 50,
            'validation_type': '5_cv1',
            'task_type': self.task_type,
            'threads': 32,
            'num_workers': -1,
            'prefetch_factor': 4,
            'chunk_size': 0,
            'similarity_threshold': 0.5,
            'save_datasets': False,
            'save_format': 'npy',
            'save_dir_prefix': 'result/data',
            'shutdown': False,
            'adv_mode': 'none',
            'adv_norm': 'linf',
            'adv_eps': 0.01,
            'adv_alpha': 0.005,
            'adv_steps': 0,
            'adv_rand_init': False,
            'adv_project': True,
            'adv_agg': 'mean',
            'adv_budget': 'shared',
            'adv_use_amp': False,
            'adv_on_moco': False,
            'adv_clip_min': float("-inf"),
            'adv_clip_max': float("inf"),
            'adv_warmup_end': 3,
            'enable_threshold_scan': True,
            'threshold_min': 0.35,
            'threshold_max': 0.65,
            'threshold_step': 0.01,
            'enable_temp_scaling': True,
            'temp_grid_min': 0.5,
            'temp_grid_max': 3.0,
            'temp_grid_num': 26,
            'kfold_recompute': True,
            'kfold_cache': False
        }
    
    def convert_to_experiment_config(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        将优化参数转换为实验配置
        
        Args:
            parameters: 优化参数字典
            
        Returns:
            实验配置字典
        """
        config = self.base_config.copy()
        
        # 直接映射的参数
        direct_mappings = {
            'dimensions': 'dimensions',
            'hidden1': 'hidden1', 
            'hidden2': 'hidden2',
            'decoder1': 'decoder1',
            'lr': 'lr',
            'dropout': 'dropout',
            'weight_decay': 'weight_decay',
            'alpha': 'alpha',
            'beta': 'beta', 
            'gamma': 'gamma',
            'gat_heads': 'gat_heads',
            'gt_heads': 'gt_heads',
            'fusion_heads': 'fusion_heads',
            'batch': 'batch',
            'moco_K': 'moco_K',
            'fusion_strategy': 'fusion_strategy',
            'feature_type': 'feature_type',
            'moco_type': 'moco_type',
            # 新增MoCo参数映射
            'moco_tau1': 'moco_tau1',
            'moco_tau2': 'moco_tau2',
            'enable_view_0': 'enable_view_0'
        }
        
        # 应用直接映射，确保正确的类型转换
        for param_name, config_name in direct_mappings.items():
            if param_name in parameters:
                value = parameters[param_name]
                
                # 整数类型参数
                if param_name in ['dimensions', 'hidden1', 'hidden2', 'decoder1', 
                                'gat_heads', 'gt_heads', 'fusion_heads', 'batch', 'moco_K']:
                    config[config_name] = int(value)
                # 浮点类型参数
                elif param_name in ['lr', 'dropout', 'weight_decay', 'alpha', 'beta', 'gamma']:
                    config[config_name] = float(value)
                # 字符串类型参数
                else:
                    config[config_name] = str(value)
        
        # 特殊处理，确保类型正确
        # MoCo相关参数
        if 'moco_K' in parameters:
            config['moco_queue'] = int(parameters['moco_K'])
        
        # 新增MoCo参数处理
        if 'moco_tau1' in parameters:
            config['moco_tau1'] = float(parameters['moco_tau1'])
        
        if 'moco_tau2' in parameters:
            config['moco_tau2'] = float(parameters['moco_tau2'])
        
        # enable_view_0字符串到布尔值转换
        if 'enable_view_0' in parameters:
            config['enable_view_0'] = str(parameters['enable_view_0']).lower() == 'true'
        
        # 参数别名处理逻辑
        # 处理moco_m别名（映射到moco_momentum）
        if 'moco_m' in parameters:
            config['moco_momentum'] = float(parameters['moco_m'])
        elif 'moco_momentum' in parameters:
            config['moco_momentum'] = float(parameters['moco_momentum'])
        
        # 处理moco_T别名（映射到moco_t）
        if 'moco_T' in parameters:
            config['moco_t'] = float(parameters['moco_T'])
        elif 'moco_t' in parameters:
            config['moco_t'] = float(parameters['moco_t'])
        
        # 损失权重别名
        if 'alpha' in parameters:
            config['loss_ratio1'] = float(parameters['alpha'])
        if 'beta' in parameters:
            config['loss_ratio2'] = float(parameters['beta'])
        if 'gamma' in parameters:
            config['loss_ratio3'] = float(parameters['gamma'])
        
        # 投影维度默认跟随hidden2
        if 'hidden2' in parameters:
            config['proj_dim'] = int(parameters['hidden2'])
        
        # 根据任务类型设置数据文件
        if self.task_type == "LDA":
            config['in_file'] = "dataset1/LDA.edgelist"
            config['neg_sample'] = "dataset1/non_LDA.edgelist"
        elif self.task_type == "MDA":
            config['in_file'] = "dataset2/MDA.edgelist"
            config['neg_sample'] = "dataset2/non_MDA.edgelist"
        elif self.task_type == "LMI":
            config['in_file'] = "dataset1/LMI.edgelist"
            config['neg_sample'] = "dataset1/non_LMI.edgelist"
        
        return config
    
    def validate_experiment_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证实验配置的完整性"""
        errors = []
        
        # 检查必需的参数
        required_params = [
            'dimensions', 'hidden1', 'hidden2', 'decoder1',
            'lr', 'dropout', 'weight_decay', 'batch',
            'gat_heads', 'gt_heads', 'fusion_heads',
            'alpha', 'beta', 'gamma', 'task_type'
        ]
        
        for param in required_params:
            if param not in config:
                errors.append(f"缺少必需参数: {param}")
        
        # 检查参数类型
        type_checks = {
            'dimensions': int,
            'hidden1': int,
            'hidden2': int,
            'decoder1': int,
            'lr': (int, float),
            'dropout': (int, float),
            'weight_decay': (int, float),
            'batch': int,
            'gat_heads': int,
            'gt_heads': int,
            'fusion_heads': int,
            'alpha': (int, float),
            'beta': (int, float),
            'gamma': (int, float)
        }
        
        for param, expected_type in type_checks.items():
            if param in config and not isinstance(config[param], expected_type):
                errors.append(f"参数 {param} 类型错误，期望 {expected_type}")
        
        # 新增MoCo参数检查
        moco_type_checks = {
            'moco_K': int,
            'moco_queue': int,
            'moco_momentum': (int, float),
            'moco_t': (int, float),
            'moco_tau1': (int, float),
            'moco_tau2': (int, float),
            'enable_view_0': bool,
            'moco_type': str
        }
        
        for param, expected_type in moco_type_checks.items():
            if param in config and not isinstance(config[param], expected_type):
                errors.append(f"MoCo参数 {param} 类型错误，期望 {expected_type}")
        
        # MoCo参数完整性检查
        if 'moco_type' in config:
            moco_type = config['moco_type']
            
            # 基本MoCo参数检查
            basic_moco_params = ['moco_K', 'moco_momentum', 'moco_t']
            for param in basic_moco_params:
                if param not in config:
                    errors.append(f"MoCo类型为 {moco_type} 时缺少参数: {param}")
            
            # DoubleTau特定参数检查
            if moco_type == 'double_tau':
                double_tau_params = ['moco_tau1', 'moco_tau2']
                for param in double_tau_params:
                    if param not in config:
                        errors.append(f"DoubleTau MoCo类型缺少参数: {param}")
                
                # 检查tau1和tau2的关系
                if all(param in config for param in ['moco_tau1', 'moco_tau2']):
                    tau1 = float(config['moco_tau1'])
                    tau2 = float(config['moco_tau2'])
                    if tau2 < tau1:
                        errors.append(f"DoubleTau MoCo参数错误: tau2 ({tau2}) 应该大于等于 tau1 ({tau1})")
        
        # MoCo参数值范围检查
        if 'moco_momentum' in config:
            momentum = float(config['moco_momentum'])
            if not (0.9 <= momentum <= 0.9999):
                errors.append(f"MoCo动量系数 {momentum} 超出合理范围 [0.9, 0.9999]")
        
        # 温度参数正值检查
        temp_params = ['moco_t', 'moco_tau1', 'moco_tau2']
        for param in temp_params:
            if param in config:
                temp = float(config[param])
                if temp <= 0:
                    errors.append(f"MoCo温度参数 {param} ({temp}) 必须为正值")
        
        # 队列大小一致性检查
        if all(param in config for param in ['moco_K', 'moco_queue']):
            if config['moco_K'] != config['moco_queue']:
                errors.append(f"MoCo队列参数不一致: moco_K ({config['moco_K']}) != moco_queue ({config['moco_queue']})")
        
        return len(errors) == 0, errors


if __name__ == "__main__":
    # 测试参数验证器
    from autodl_core import create_default_parameter_space
    
    print("测试参数验证器...")
    
    # 创建参数空间和验证器
    space = create_default_parameter_space()
    validator = ParameterValidator(space)
    
    # 测试有效参数
    valid_params = {
        'dimensions': 256,
        'hidden1': 128,
        'hidden2': 64,
        'decoder1': 512,
        'lr': 0.001,
        'dropout': 0.1,
        'weight_decay': 0.0005,
        'alpha': 1.0,
        'beta': 0.5,
        'gamma': 0.5,
        'gat_heads': 4,
        'gt_heads': 4,
        'fusion_heads': 4,
        'batch': 32,
        'moco_K': 4096,
        'fusion_strategy': 'self_attention',
        'feature_type': 'normal',
        'moco_type': 'basic'
    }
    
    is_valid, errors = validator.validate_parameters(valid_params)
    print(f"有效参数验证: {is_valid}, 错误: {errors}")
    
    # 测试无效参数（违反约束）
    invalid_params = valid_params.copy()
    invalid_params['hidden1'] = 300  # 违反递减约束
    invalid_params['gat_heads'] = 3   # 违反整除约束
    
    is_valid, errors = validator.validate_parameters(invalid_params)
    print(f"无效参数验证: {is_valid}, 错误: {errors}")
    
    # 测试参数修复
    fixed_params = validator.suggest_parameter_fix(invalid_params)
    print(f"修复后参数: hidden1={fixed_params['hidden1']}, gat_heads={fixed_params['gat_heads']}")
    
    # 测试配置转换器
    converter = ConfigurationConverter("LDA")
    exp_config = converter.convert_to_experiment_config(valid_params)
    print(f"实验配置转换完成，包含 {len(exp_config)} 个配置项")
    
    # 验证实验配置
    config_valid, config_errors = converter.validate_experiment_config(exp_config)
    print(f"实验配置验证: {config_valid}, 错误: {config_errors}")
    
    print("参数验证器测试完成!")