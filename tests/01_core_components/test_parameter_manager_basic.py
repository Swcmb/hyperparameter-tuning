#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
参数管理器基础测试

测试核心功能是否正常工作
"""

import sys
from parameter_manager import (
    ParameterManager, ParameterDefinition, LazyParameterProxy,
    get_parameter_manager, register_module_parameters, ParameterManagerError
)


def test_singleton_behavior():
    """测试单例模式"""
    manager1 = ParameterManager()
    manager2 = ParameterManager()
    assert manager1 is manager2, "ParameterManager应该是单例"


def test_parameter_definition():
    """测试参数定义"""
    param_def = ParameterDefinition(
        name="test_param",
        type=int,
        default=42,
        help="测试参数"
    )
    
    # 测试验证
    assert param_def.validate(42) == 42
    assert param_def.validate("42") == 42  # 类型转换
    
    # 测试选择验证
    param_def.choices = [1, 2, 3]
    assert param_def.validate(2) == 2
    
    try:
        param_def.validate(5)  # 不在选择范围内
        assert False, "应该抛出异常"
    except ParameterManagerError:
        pass  # 预期的异常


def test_module_registration():
    """测试模块参数注册"""
    manager = ParameterManager()
    
    # 注册autodl模块参数
    autodl_params = [
        ParameterDefinition(
            name="max_iterations",
            type=int,
            default=50,
            help="最大迭代次数"
        ),
        ParameterDefinition(
            name="random_seed",
            type=int,
            default=42,
            help="随机种子"
        )
    ]
    
    manager.register_module_parser("autodl", autodl_params)
    
    # 注册parms_setting模块参数
    parms_params = [
        ParameterDefinition(
            name="seed",
            type=int,
            default=0,
            help="随机种子"
        ),
        ParameterDefinition(
            name="epochs",
            type=int,
            default=50,
            help="训练轮数"
        )
    ]
    
    manager.register_module_parser("parms_setting", parms_params)
    
    # 检查合并后的定义
    merged = manager._registry.get_merged_definitions()
    assert "max_iterations" in merged
    assert "random_seed" in merged
    assert "seed" in merged
    assert "epochs" in merged


def test_parameter_parsing():
    """测试参数解析"""
    manager = ParameterManager()
    
    # 注册测试参数
    test_params = [
        ParameterDefinition(
            name="test_int",
            type=int,
            default=10,
            help="测试整数"
        ),
        ParameterDefinition(
            name="test_str",
            type=str,
            default="hello",
            help="测试字符串"
        )
    ]
    
    manager.register_module_parser("test", test_params)
    
    # 模拟命令行参数
    test_args = ["--test_int", "20", "--test_str", "world"]
    
    # 解析参数
    parsed = manager.parse_arguments(test_args)
    
    assert parsed.test_int == 20
    assert parsed.test_str == "world"


def test_lazy_proxy():
    """测试延迟代理"""
    # 重置单例状态进行测试
    ParameterManager._instance = None
    ParameterManager._initialized = False
    
    manager = ParameterManager()
    
    # 注册参数
    params = [
        ParameterDefinition(
            name="proxy_test",
            type=str,
            default="default_value",
            help="代理测试"
        )
    ]
    
    manager.register_module_parser("proxy", params)
    
    # 创建代理
    proxy = LazyParameterProxy(manager)
    
    # 测试延迟访问
    value = proxy.get("proxy_test")
    print(f"代理测试值: {value}")
    assert value == "default_value", f"期望 'default_value', 得到 {value}"


if __name__ == "__main__":
    # 运行基础测试
    test_singleton_behavior()
    print("✓ 单例模式测试通过")
    
    test_parameter_definition()
    print("✓ 参数定义测试通过")
    
    test_module_registration()
    print("✓ 模块注册测试通过")
    
    test_parameter_parsing()
    print("✓ 参数解析测试通过")
    
    test_lazy_proxy()
    print("✓ 延迟代理测试通过")
    
    print("\n🎉 所有基础测试通过！")