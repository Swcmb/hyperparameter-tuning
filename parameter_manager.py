#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一参数管理系统

本模块实现了一个统一的参数管理系统，解决多模块间的命令行参数冲突问题。
采用单例模式的参数管理器，结合延迟初始化和模块参数注册机制。

主要组件：
- ParameterManager: 单例参数管理器
- ParameterDefinition: 参数定义数据模型
- ParsedParameter: 解析后的参数数据模型
- ModuleParameterRegistry: 模块参数注册表
- LazyParameterProxy: 延迟参数代理
"""

import argparse
import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
import json


class ParameterManagerError(Exception):
    """参数管理器相关错误"""
    pass


@dataclass
class ParameterDefinition:
    """参数定义数据模型"""
    name: str
    type: type
    default: Any
    help: str
    required: bool = False
    choices: Optional[List[Any]] = None
    module: str = None
    aliases: Optional[List[str]] = None
    validator: Optional[Callable[[Any], Any]] = None
    dest: Optional[str] = None  # 目标属性名，用于别名支持
    
    def validate(self, value: Any) -> Any:
        """验证参数值"""
        if self.validator:
            return self.validator(value)
        
        # 基本类型验证
        if value is not None and self.type != type(value):
            try:
                # 尝试类型转换
                if self.type == bool and isinstance(value, str):
                    return str(value).lower() in ('true', '1', 'yes', 'on')
                else:
                    return self.type(value)
            except (ValueError, TypeError) as e:
                raise ParameterManagerError(
                    f"参数 {self.name} 类型错误: 期望 {self.type.__name__}, 得到 {type(value).__name__}"
                )
        
        # 选择验证
        if self.choices and value not in self.choices:
            raise ParameterManagerError(
                f"参数 {self.name} 值无效: {value}, 可选值: {self.choices}"
            )
        
        return value


@dataclass
class ParsedParameter:
    """解析后的参数数据模型"""
    name: str
    value: Any
    source: str  # 'command_line', 'config_file', 'default', 'environment'
    module: str
    definition: ParameterDefinition
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'name': self.name,
            'value': self.value,
            'source': self.source,
            'module': self.module,
            'type': self.definition.type.__name__,
            'help': self.definition.help
        }


@dataclass
class ParameterConflict:
    """参数冲突数据模型"""
    parameter_name: str
    conflicting_modules: List[str]
    conflict_type: str  # 'name', 'type', 'default'
    resolution_strategy: str = 'last_wins'
    
    def resolve(self, definitions: List[ParameterDefinition]) -> ParameterDefinition:
        """解决冲突并返回最终定义"""
        if self.resolution_strategy == 'last_wins':
            return definitions[-1]  # 使用最后注册的定义
        elif self.resolution_strategy == 'most_strict':
            # 使用最严格的类型定义
            return max(definitions, key=lambda d: len(d.choices) if d.choices else 0)
        else:
            return definitions[0]  # 默认使用第一个


class ModuleParameterRegistry:
    """模块参数注册表"""
    
    def __init__(self):
        self._module_definitions: Dict[str, List[ParameterDefinition]] = {}
        self._parameter_conflicts: Dict[str, ParameterConflict] = {}
        self._merged_definitions: Optional[Dict[str, ParameterDefinition]] = None
        self._lock = threading.Lock()
    
    def register_parameters(self, module_name: str, parameters: List[ParameterDefinition]):
        """注册模块参数定义"""
        with self._lock:
            if module_name in self._module_definitions:
                logging.warning(f"模块 {module_name} 的参数已存在，将被覆盖")
            
            # 设置模块名
            for param in parameters:
                param.module = module_name
            
            self._module_definitions[module_name] = parameters
            self._merged_definitions = None  # 清除缓存，强制重新合并
            
            # 检测冲突
            self._detect_conflicts()
    
    def _detect_conflicts(self):
        """检测参数冲突"""
        self._parameter_conflicts.clear()
        param_to_modules = {}
        
        # 收集所有参数名和对应的模块
        for module_name, params in self._module_definitions.items():
            for param in params:
                param_name = param.name
                if param_name not in param_to_modules:
                    param_to_modules[param_name] = []
                param_to_modules[param_name].append((module_name, param))
        
        # 检测冲突
        for param_name, module_params in param_to_modules.items():
            if len(module_params) > 1:
                modules = [mp[0] for mp in module_params]
                params = [mp[1] for mp in module_params]
                
                # 检查类型冲突
                types = set(p.type for p in params)
                defaults = set(p.default for p in params)
                
                conflict_type = 'name'
                if len(types) > 1:
                    conflict_type = 'type'
                elif len(defaults) > 1:
                    conflict_type = 'default'
                
                self._parameter_conflicts[param_name] = ParameterConflict(
                    parameter_name=param_name,
                    conflicting_modules=modules,
                    conflict_type=conflict_type
                )
    
    def get_merged_definitions(self) -> Dict[str, ParameterDefinition]:
        """获取合并后的参数定义"""
        if self._merged_definitions is None:
            self._merged_definitions = self._merge_definitions()
        return self._merged_definitions
    
    def _merge_definitions(self) -> Dict[str, ParameterDefinition]:
        """合并所有模块的参数定义"""
        merged = {}
        
        for module_name, params in self._module_definitions.items():
            for param in params:
                param_name = param.name
                
                if param_name in self._parameter_conflicts:
                    # 解决冲突
                    conflict = self._parameter_conflicts[param_name]
                    all_params = []
                    for mod_name, mod_params in self._module_definitions.items():
                        for p in mod_params:
                            if p.name == param_name:
                                all_params.append(p)
                    
                    resolved_param = conflict.resolve(all_params)
                    merged[param_name] = resolved_param
                else:
                    merged[param_name] = param
        
        return merged
    
    def get_conflicts(self) -> Dict[str, ParameterConflict]:
        """获取参数冲突信息"""
        return self._parameter_conflicts.copy()


class ParameterManager:
    """统一参数管理器 - 单例模式"""
    
    _instance = None
    _initialized = False
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._registry = ModuleParameterRegistry()
                    self._parsed_args = None
                    self._raw_args = None
                    self._config_sources = []
                    self._logger = logging.getLogger(__name__)
                    self._parse_lock = threading.Lock()
                    ParameterManager._initialized = True
    
    def register_module_parser(self, module_name: str, parameters: List[ParameterDefinition]):
        """注册模块的参数定义"""
        self._logger.info(f"注册模块 {module_name} 的 {len(parameters)} 个参数")
        self._registry.register_parameters(module_name, parameters)
    
    def get_parameter(self, key: str, default=None, module: str = None) -> Any:
        """获取参数值 - 支持延迟解析"""
        if self._parsed_args is None:
            self._ensure_parsed()
        
        if hasattr(self._parsed_args, key):
            return getattr(self._parsed_args, key)
        else:
            return default
    
    def _ensure_parsed(self):
        """确保参数已解析 - 延迟初始化"""
        if self._parsed_args is None:
            with self._parse_lock:
                if self._parsed_args is None:
                    self.parse_arguments()
    
    def parse_arguments(self, args=None):
        """执行参数解析 - 只执行一次"""
        if self._parsed_args is not None:
            self._logger.warning("参数已解析，跳过重复解析")
            return self._parsed_args
        
        try:
            parser = self._create_merged_parser()
            
            # 使用提供的参数或系统参数
            if args is None:
                args = sys.argv[1:]
            
            self._raw_args = args
            self._parsed_args = parser.parse_args(args)
            
            self._logger.info(f"成功解析 {len(vars(self._parsed_args))} 个参数")
            return self._parsed_args
            
        except SystemExit as e:
            if e.code != 0:
                self._logger.error("参数解析失败")
                raise ParameterManagerError("命令行参数解析失败")
            raise
        except Exception as e:
            self._logger.error(f"参数解析时发生错误: {e}")
            raise ParameterManagerError(f"参数解析错误: {e}")
    
    def _create_merged_parser(self) -> argparse.ArgumentParser:
        """创建合并后的参数解析器"""
        parser = argparse.ArgumentParser(
            description="统一参数管理系统",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        merged_definitions = self._registry.get_merged_definitions()
        
        for param_name, param_def in merged_definitions.items():
            kwargs = {
                'type': param_def.type,
                'default': param_def.default,
                'help': param_def.help
            }
            
            if param_def.required:
                kwargs['required'] = True
            
            if param_def.choices:
                kwargs['choices'] = param_def.choices
            
            if param_def.dest:
                kwargs['dest'] = param_def.dest
            
            # 处理布尔类型的特殊情况
            if param_def.type == bool:
                kwargs['type'] = lambda x: str(x).lower() in ('true', '1', 'yes', 'on')
            
            parser.add_argument(f'--{param_name}', **kwargs)
        
        return parser
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """获取所有解析后的参数"""
        if self._parsed_args is None:
            self._ensure_parsed()
        
        return vars(self._parsed_args)
    
    def get_conflicts(self) -> Dict[str, ParameterConflict]:
        """获取参数冲突信息"""
        return self._registry.get_conflicts()
    
    def add_config_source(self, source_path: str, format: str = 'auto'):
        """添加配置文件来源"""
        # TODO: 在后续任务中实现配置文件支持
        self._config_sources.append((source_path, format))
        self._logger.info(f"添加配置文件来源: {source_path}")


class LazyParameterProxy:
    """延迟参数代理 - 保持向后兼容性"""
    
    def __init__(self, manager: Optional[ParameterManager] = None):
        self._manager = manager or ParameterManager()
        self._cache = {}
    
    def __getattr__(self, name):
        """动态属性访问 - 触发延迟解析"""
        if name.startswith('_'):
            # 避免无限递归
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if name not in self._cache:
            self._cache[name] = self._manager.get_parameter(name)
        return self._cache[name]
    
    def get_args(self):
        """兼容 layer.py 的 get_args() 接口"""
        return self._manager._parsed_args or self._manager.parse_arguments()
    
    def settings(self):
        """兼容 parms_setting.py 的 settings() 接口"""
        return self.get_args()
    
    def get(self, key: str, default=None):
        """获取参数值"""
        return self._manager.get_parameter(key, default)


# 全局实例
_global_manager = None
_global_proxy = None


def get_parameter_manager() -> ParameterManager:
    """获取全局参数管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ParameterManager()
    return _global_manager


def get_parameter_proxy() -> LazyParameterProxy:
    """获取全局参数代理实例"""
    global _global_proxy
    if _global_proxy is None:
        _global_proxy = LazyParameterProxy(get_parameter_manager())
    return _global_proxy


# 便捷函数
def register_module_parameters(module_name: str, parameters: List[ParameterDefinition]):
    """注册模块参数的便捷函数"""
    manager = get_parameter_manager()
    manager.register_module_parser(module_name, parameters)


def get_parameter(key: str, default=None) -> Any:
    """获取参数值的便捷函数"""
    proxy = get_parameter_proxy()
    return proxy.get(key, default)