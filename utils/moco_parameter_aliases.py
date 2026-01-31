"""
MoCo参数别名映射配置

本模块定义了MoCo参数的别名映射关系，用于处理参数统一过程中的向后兼容性。
"""

from typing import Dict, Any

# MoCo参数别名映射表
# 格式: {别名: 标准参数名}
MOCO_PARAMETER_ALIASES = {
    # 动量系数别名
    'moco_m': 'moco_momentum',
    
    # 温度系数别名
    'moco_T': 'moco_t',
    
    # 可能的未来别名扩展
    'momentum': 'moco_momentum',
    'temperature': 'moco_t',
    'tau': 'moco_t',
    'queue_size': 'moco_K',
    'queue_length': 'moco_queue'
}

# 反向映射表（标准参数名到所有可能的别名）
MOCO_PARAMETER_REVERSE_ALIASES = {}
for alias, standard in MOCO_PARAMETER_ALIASES.items():
    if standard not in MOCO_PARAMETER_REVERSE_ALIASES:
        MOCO_PARAMETER_REVERSE_ALIASES[standard] = []
    MOCO_PARAMETER_REVERSE_ALIASES[standard].append(alias)

# MoCo参数默认值配置
MOCO_PARAMETER_DEFAULTS = {
    'moco_momentum': 0.999,
    'moco_t': 0.2,
    'moco_tau1': 0.2,
    'moco_tau2': 0.3,
    'moco_K': 4096,
    'moco_queue': 4096,
    'enable_view_0': True,
    'moco_type': 'basic',
    'proj_dim': None,
    'queue_warmup_steps': 0
}

# MoCo参数类型定义
MOCO_PARAMETER_TYPES = {
    'moco_momentum': float,
    'moco_t': float,
    'moco_tau1': float,
    'moco_tau2': float,
    'moco_K': int,
    'moco_queue': int,
    'enable_view_0': bool,
    'moco_type': str,
    'proj_dim': int,
    'queue_warmup_steps': int
}

# MoCo参数范围定义
MOCO_PARAMETER_RANGES = {
    'moco_momentum': (0.9, 0.9999),
    'moco_t': (0.01, 1.0),
    'moco_tau1': (0.01, 1.0),
    'moco_tau2': (0.01, 1.0),
    'moco_K': [1024, 2048, 4096, 8192],
    'moco_queue': [1024, 2048, 4096, 8192],
    'enable_view_0': ['true', 'false'],
    'moco_type': ['basic', 'double_tau'],
    'proj_dim': (32, 512),
    'queue_warmup_steps': (0, 1000)
}


def resolve_parameter_alias(param_name: str) -> str:
    """
    解析参数别名，返回标准参数名
    
    Args:
        param_name: 参数名（可能是别名）
        
    Returns:
        标准参数名
    """
    return MOCO_PARAMETER_ALIASES.get(param_name, param_name)


def apply_parameter_aliases(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    应用参数别名映射，将别名转换为标准参数名
    
    Args:
        parameters: 原始参数字典
        
    Returns:
        转换后的参数字典
    """
    resolved_params = {}
    
    for param_name, value in parameters.items():
        standard_name = resolve_parameter_alias(param_name)
        resolved_params[standard_name] = value
    
    return resolved_params


def get_parameter_default(param_name: str) -> Any:
    """
    获取参数的默认值
    
    Args:
        param_name: 参数名
        
    Returns:
        参数默认值，如果参数不存在则返回None
    """
    standard_name = resolve_parameter_alias(param_name)
    return MOCO_PARAMETER_DEFAULTS.get(standard_name)


def get_parameter_type(param_name: str) -> type:
    """
    获取参数的类型
    
    Args:
        param_name: 参数名
        
    Returns:
        参数类型，如果参数不存在则返回None
    """
    standard_name = resolve_parameter_alias(param_name)
    return MOCO_PARAMETER_TYPES.get(standard_name)


def get_parameter_range(param_name: str) -> Any:
    """
    获取参数的取值范围
    
    Args:
        param_name: 参数名
        
    Returns:
        参数取值范围（元组表示连续范围，列表表示离散值）
    """
    standard_name = resolve_parameter_alias(param_name)
    return MOCO_PARAMETER_RANGES.get(standard_name)


def validate_moco_parameter(param_name: str, value: Any) -> bool:
    """
    验证MoCo参数值是否有效
    
    Args:
        param_name: 参数名
        value: 参数值
        
    Returns:
        True如果参数值有效，否则False
    """
    standard_name = resolve_parameter_alias(param_name)
    param_range = get_parameter_range(standard_name)
    param_type = get_parameter_type(standard_name)
    
    if param_range is None or param_type is None:
        return False
    
    try:
        # 类型转换验证
        if param_type == bool:
            if isinstance(value, str):
                value = value.lower() in ['true', '1', 'yes', 'on']
            else:
                value = bool(value)
        else:
            value = param_type(value)
        
        # 范围验证
        if isinstance(param_range, tuple):
            # 连续范围
            return param_range[0] <= value <= param_range[1]
        elif isinstance(param_range, list):
            # 离散值
            return value in param_range
        
    except (ValueError, TypeError):
        return False
    
    return False


def get_all_moco_parameters() -> list:
    """
    获取所有MoCo参数的标准名称列表
    
    Returns:
        MoCo参数名称列表
    """
    return list(MOCO_PARAMETER_DEFAULTS.keys())


def is_moco_parameter(param_name: str) -> bool:
    """
    检查参数是否为MoCo相关参数
    
    Args:
        param_name: 参数名
        
    Returns:
        True如果是MoCo参数，否则False
    """
    standard_name = resolve_parameter_alias(param_name)
    return standard_name in MOCO_PARAMETER_DEFAULTS


if __name__ == "__main__":
    # 测试代码
    print("测试MoCo参数别名映射...")
    
    # 测试别名解析
    test_params = {
        'moco_m': 0.999,
        'moco_T': 0.2,
        'moco_tau1': 0.15,
        'enable_view_0': 'true'
    }
    
    print(f"原始参数: {test_params}")
    resolved = apply_parameter_aliases(test_params)
    print(f"解析后参数: {resolved}")
    
    # 测试参数验证
    for param, value in resolved.items():
        is_valid = validate_moco_parameter(param, value)
        print(f"参数 {param}={value} 验证结果: {is_valid}")
    
    # 测试默认值获取
    print(f"\n默认值测试:")
    for param in get_all_moco_parameters():
        default = get_parameter_default(param)
        param_type = get_parameter_type(param)
        param_range = get_parameter_range(param)
        print(f"  {param}: 默认={default}, 类型={param_type.__name__}, 范围={param_range}")
    
    print("\nMoCo参数别名映射测试完成!")