"""
结构化标签处理器 (StructuredTagProcessor)

处理和验证结构化标签的组件，定义所有预设的标签类别和验证规则。
实现标签验证和格式化方法，创建标签到颜色/样式的映射系统。

主要功能：
- 预定义的标签类别管理
- 标签验证和格式化
- 标签层次结构管理
- 颜色和样式映射
- 标签使用统计
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import re
import threading
from collections import defaultdict, Counter


class TagCategory(Enum):
    """标签类别枚举"""
    TRAINING = "training"
    CONFIG = "config"
    OPTIMIZATION = "optimization"
    SYSTEM = "system"
    STATUS = "status"
    RESULTS = "results"
    PROGRESS = "progress"
    DATA = "data"
    MODEL = "model"
    EVALUATION = "evaluation"


class TagLevel(Enum):
    """标签级别枚举"""
    PRIMARY = "primary"      # 主要标签，如 TRAINING, OPTIMIZATION
    SECONDARY = "secondary"  # 次要标签，如 EPOCH, BATCH
    DETAIL = "detail"        # 详细标签，如 LOSS, METRICS
    CONTEXT = "context"      # 上下文标签，如 GPU, MEMORY


@dataclass
class TagInfo:
    """标签信息数据结构"""
    name: str
    category: TagCategory
    level: TagLevel
    description: str
    parent_tags: List[str] = None
    child_tags: List[str] = None
    aliases: List[str] = None
    color_code: str = None
    usage_count: int = 0
    
    def __post_init__(self):
        if self.parent_tags is None:
            self.parent_tags = []
        if self.child_tags is None:
            self.child_tags = []
        if self.aliases is None:
            self.aliases = []


class StructuredTagProcessor:
    """
    结构化标签处理器
    
    处理和验证结构化标签，提供标签管理和格式化功能
    """
    
    def __init__(self):
        """初始化结构化标签处理器"""
        self._tags: Dict[str, TagInfo] = {}
        self._category_tags: Dict[TagCategory, Set[str]] = defaultdict(set)
        self._level_tags: Dict[TagLevel, Set[str]] = defaultdict(set)
        self._alias_mapping: Dict[str, str] = {}
        self._usage_stats: Counter = Counter()
        self._lock = threading.RLock()
        
        # 初始化预定义标签
        self._initialize_predefined_tags()
    
    def _initialize_predefined_tags(self):
        """初始化预定义的标签"""
        
        # 训练相关标签
        training_tags = [
            TagInfo("TRAINING", TagCategory.TRAINING, TagLevel.PRIMARY, 
                   "训练过程的主要标签", color_code="blue"),
            TagInfo("EPOCH", TagCategory.TRAINING, TagLevel.SECONDARY, 
                   "训练轮次标签", parent_tags=["TRAINING"], color_code="cyan"),
            TagInfo("BATCH", TagCategory.TRAINING, TagLevel.DETAIL, 
                   "批次处理标签", parent_tags=["TRAINING", "EPOCH"], color_code="light_blue"),
            TagInfo("LOSS", TagCategory.TRAINING, TagLevel.DETAIL, 
                   "损失函数标签", parent_tags=["TRAINING", "EPOCH", "BATCH"], color_code="red"),
            TagInfo("METRICS", TagCategory.TRAINING, TagLevel.DETAIL, 
                   "性能指标标签", parent_tags=["TRAINING", "EPOCH"], color_code="green"),
            TagInfo("CONVERGENCE", TagCategory.TRAINING, TagLevel.SECONDARY, 
                   "收敛检测标签", parent_tags=["TRAINING"], color_code="purple"),
        ]
        
        # 配置相关标签
        config_tags = [
            TagInfo("CONFIG", TagCategory.CONFIG, TagLevel.PRIMARY, 
                   "配置信息的主要标签", color_code="yellow"),
            TagInfo("PARAMS", TagCategory.CONFIG, TagLevel.SECONDARY, 
                   "参数配置标签", parent_tags=["CONFIG"], color_code="orange"),
            TagInfo("SETUP", TagCategory.CONFIG, TagLevel.SECONDARY, 
                   "设置配置标签", parent_tags=["CONFIG"], color_code="brown"),
            TagInfo("INIT", TagCategory.CONFIG, TagLevel.SECONDARY, 
                   "初始化标签", parent_tags=["CONFIG"], color_code="magenta"),
            TagInfo("VALIDATION", TagCategory.CONFIG, TagLevel.DETAIL, 
                   "配置验证标签", parent_tags=["CONFIG"], color_code="pink"),
        ]
        
        # 优化相关标签
        optimization_tags = [
            TagInfo("OPTIMIZATION", TagCategory.OPTIMIZATION, TagLevel.PRIMARY, 
                   "优化过程的主要标签", color_code="green"),
            TagInfo("ACQUISITION", TagCategory.OPTIMIZATION, TagLevel.SECONDARY, 
                   "采集函数标签", parent_tags=["OPTIMIZATION"], color_code="lime"),
            TagInfo("SUGGESTION", TagCategory.OPTIMIZATION, TagLevel.SECONDARY, 
                   "参数建议标签", parent_tags=["OPTIMIZATION"], color_code="olive"),
            TagInfo("GP_UPDATE", TagCategory.OPTIMIZATION, TagLevel.DETAIL, 
                   "高斯过程更新标签", parent_tags=["OPTIMIZATION"], color_code="teal"),
            TagInfo("PARETO", TagCategory.OPTIMIZATION, TagLevel.DETAIL, 
                   "帕累托前沿标签", parent_tags=["OPTIMIZATION"], color_code="navy"),
        ]
        
        # 系统相关标签
        system_tags = [
            TagInfo("SYSTEM", TagCategory.SYSTEM, TagLevel.PRIMARY, 
                   "系统信息的主要标签", color_code="gray"),
            TagInfo("MEMORY", TagCategory.SYSTEM, TagLevel.SECONDARY, 
                   "内存使用标签", parent_tags=["SYSTEM"], color_code="silver"),
            TagInfo("GPU", TagCategory.SYSTEM, TagLevel.SECONDARY, 
                   "GPU状态标签", parent_tags=["SYSTEM"], color_code="gold"),
            TagInfo("CPU", TagCategory.SYSTEM, TagLevel.SECONDARY, 
                   "CPU状态标签", parent_tags=["SYSTEM"], color_code="bronze"),
            TagInfo("DISK", TagCategory.SYSTEM, TagLevel.SECONDARY, 
                   "磁盘状态标签", parent_tags=["SYSTEM"], color_code="maroon"),
            TagInfo("DEVICE", TagCategory.SYSTEM, TagLevel.CONTEXT, 
                   "设备信息标签", parent_tags=["SYSTEM"], color_code="indigo"),
        ]
        
        # 状态相关标签
        status_tags = [
            TagInfo("ERROR", TagCategory.STATUS, TagLevel.PRIMARY, 
                   "错误信息标签", color_code="red"),
            TagInfo("WARNING", TagCategory.STATUS, TagLevel.PRIMARY, 
                   "警告信息标签", color_code="orange"),
            TagInfo("INFO", TagCategory.STATUS, TagLevel.PRIMARY, 
                   "信息标签", color_code="blue"),
            TagInfo("DEBUG", TagCategory.STATUS, TagLevel.PRIMARY, 
                   "调试信息标签", color_code="gray"),
            TagInfo("SUCCESS", TagCategory.STATUS, TagLevel.PRIMARY, 
                   "成功信息标签", color_code="green"),
            TagInfo("CRITICAL", TagCategory.STATUS, TagLevel.PRIMARY, 
                   "严重错误标签", color_code="dark_red"),
        ]
        
        # 结果相关标签
        results_tags = [
            TagInfo("RESULTS", TagCategory.RESULTS, TagLevel.PRIMARY, 
                   "结果信息的主要标签", color_code="purple"),
            TagInfo("ANALYSIS", TagCategory.RESULTS, TagLevel.SECONDARY, 
                   "分析结果标签", parent_tags=["RESULTS"], color_code="violet"),
            TagInfo("SUMMARY", TagCategory.RESULTS, TagLevel.SECONDARY, 
                   "总结信息标签", parent_tags=["RESULTS"], color_code="lavender"),
            TagInfo("REPORT", TagCategory.RESULTS, TagLevel.SECONDARY, 
                   "报告生成标签", parent_tags=["RESULTS"], color_code="plum"),
            TagInfo("VISUALIZATION", TagCategory.RESULTS, TagLevel.DETAIL, 
                   "可视化标签", parent_tags=["RESULTS"], color_code="orchid"),
        ]
        
        # 进度相关标签
        progress_tags = [
            TagInfo("PROGRESS", TagCategory.PROGRESS, TagLevel.PRIMARY, 
                   "进度信息的主要标签", color_code="cyan"),
            TagInfo("CHECKPOINT", TagCategory.PROGRESS, TagLevel.SECONDARY, 
                   "检查点标签", parent_tags=["PROGRESS"], color_code="turquoise"),
            TagInfo("MILESTONE", TagCategory.PROGRESS, TagLevel.SECONDARY, 
                   "里程碑标签", parent_tags=["PROGRESS"], color_code="aqua"),
            TagInfo("ETA", TagCategory.PROGRESS, TagLevel.DETAIL, 
                   "预计完成时间标签", parent_tags=["PROGRESS"], color_code="light_cyan"),
        ]
        
        # 数据相关标签
        data_tags = [
            TagInfo("DATA", TagCategory.DATA, TagLevel.PRIMARY, 
                   "数据处理的主要标签", color_code="brown"),
            TagInfo("DATA_STATS", TagCategory.DATA, TagLevel.SECONDARY, 
                   "数据统计标签", parent_tags=["DATA"], color_code="tan"),
            TagInfo("DATA_LOAD", TagCategory.DATA, TagLevel.SECONDARY, 
                   "数据加载标签", parent_tags=["DATA"], color_code="khaki"),
            TagInfo("DATA_PREP", TagCategory.DATA, TagLevel.SECONDARY, 
                   "数据预处理标签", parent_tags=["DATA"], color_code="wheat"),
        ]
        
        # 模型相关标签
        model_tags = [
            TagInfo("MODEL", TagCategory.MODEL, TagLevel.PRIMARY, 
                   "模型相关的主要标签", color_code="dark_blue"),
            TagInfo("MODEL_INIT", TagCategory.MODEL, TagLevel.SECONDARY, 
                   "模型初始化标签", parent_tags=["MODEL"], color_code="royal_blue"),
            TagInfo("MODEL_SAVE", TagCategory.MODEL, TagLevel.SECONDARY, 
                   "模型保存标签", parent_tags=["MODEL"], color_code="steel_blue"),
            TagInfo("MODEL_LOAD", TagCategory.MODEL, TagLevel.SECONDARY, 
                   "模型加载标签", parent_tags=["MODEL"], color_code="sky_blue"),
        ]
        
        # 评估相关标签
        evaluation_tags = [
            TagInfo("EVALUATION", TagCategory.EVALUATION, TagLevel.PRIMARY, 
                   "评估过程的主要标签", color_code="dark_green"),
            TagInfo("CROSS_VALIDATION", TagCategory.EVALUATION, TagLevel.SECONDARY, 
                   "交叉验证标签", parent_tags=["EVALUATION"], color_code="forest_green"),
            TagInfo("TEST", TagCategory.EVALUATION, TagLevel.SECONDARY, 
                   "测试评估标签", parent_tags=["EVALUATION"], color_code="lime_green"),
            TagInfo("BENCHMARK", TagCategory.EVALUATION, TagLevel.SECONDARY, 
                   "基准测试标签", parent_tags=["EVALUATION"], color_code="spring_green"),
        ]
        
        # 注册所有标签
        all_tags = (training_tags + config_tags + optimization_tags + 
                   system_tags + status_tags + results_tags + 
                   progress_tags + data_tags + model_tags + evaluation_tags)
        
        for tag_info in all_tags:
            self._register_tag(tag_info)
        
        # 设置别名
        self._setup_aliases()
    
    def _register_tag(self, tag_info: TagInfo):
        """注册标签"""
        with self._lock:
            self._tags[tag_info.name] = tag_info
            self._category_tags[tag_info.category].add(tag_info.name)
            self._level_tags[tag_info.level].add(tag_info.name)
    
    def _setup_aliases(self):
        """设置标签别名"""
        aliases = {
            # 训练相关别名
            "TRAIN": "TRAINING",
            "EP": "EPOCH",
            "BATCH_PROGRESS": "BATCH",
            "LOSS_INFO": "LOSS",
            "METRIC": "METRICS",
            "CONV": "CONVERGENCE",
            
            # 配置相关别名
            "CFG": "CONFIG",
            "PARAM": "PARAMS",
            "INITIALIZE": "INIT",
            "VALIDATE": "VALIDATION",
            
            # 优化相关别名
            "OPT": "OPTIMIZATION",
            "ACQ": "ACQUISITION",
            "SUGGEST": "SUGGESTION",
            "GP": "GP_UPDATE",
            
            # 系统相关别名
            "SYS": "SYSTEM",
            "MEM": "MEMORY",
            "CUDA": "GPU",
            "PROCESSOR": "CPU",
            "STORAGE": "DISK",
            "DEV": "DEVICE",
            
            # 状态相关别名
            "ERR": "ERROR",
            "WARN": "WARNING",
            "INFORMATION": "INFO",
            "DBG": "DEBUG",
            "OK": "SUCCESS",
            "FATAL": "CRITICAL",
            
            # 结果相关别名
            "RESULT": "RESULTS",
            "ANALYZE": "ANALYSIS",
            "SUM": "SUMMARY",
            "REP": "REPORT",
            "VIZ": "VISUALIZATION",
            
            # 进度相关别名
            "PROG": "PROGRESS",
            "CHKPT": "CHECKPOINT",
            "MILE": "MILESTONE",
            "ESTIMATE": "ETA",
            
            # 数据相关别名
            "DAT": "DATA",
            "STATS": "DATA_STATS",
            "LOAD": "DATA_LOAD",
            "PREPROCESS": "DATA_PREP",
            
            # 模型相关别名
            "MDL": "MODEL",
            "INIT_MODEL": "MODEL_INIT",
            "SAVE_MODEL": "MODEL_SAVE",
            "LOAD_MODEL": "MODEL_LOAD",
            
            # 评估相关别名
            "EVAL": "EVALUATION",
            "CV": "CROSS_VALIDATION",
            "TESTING": "TEST",
            "BENCH": "BENCHMARK",
        }
        
        with self._lock:
            self._alias_mapping.update(aliases)
    
    def validate_tag(self, tag: str) -> bool:
        """
        验证标签是否有效
        
        Args:
            tag: 要验证的标签
            
        Returns:
            bool: 标签是否有效
        """
        if not tag:
            return False
        
        # 标准化标签名称
        normalized_tag = self._normalize_tag_name(tag)
        
        with self._lock:
            # 检查直接匹配
            if normalized_tag in self._tags:
                return True
            
            # 检查别名匹配
            if normalized_tag in self._alias_mapping:
                return True
            
            # 检查模式匹配（如 EPOCH_01, BATCH_001 等）
            return self._validate_pattern_tag(normalized_tag)
    
    def _normalize_tag_name(self, tag: str) -> str:
        """标准化标签名称"""
        if not tag:
            return ""
        
        # 转换为大写并移除多余空格
        normalized = tag.strip().upper()
        
        # 移除方括号
        normalized = normalized.strip('[]')
        
        # 替换常见分隔符为下划线
        normalized = re.sub(r'[-\s\.]+', '_', normalized)
        
        return normalized
    
    def _validate_pattern_tag(self, tag: str) -> bool:
        """验证模式标签（如带数字后缀的标签）"""
        # 匹配带数字后缀的标签模式
        pattern_match = re.match(r'^([A-Z_]+)_(\d+)$', tag)
        if pattern_match:
            base_tag = pattern_match.group(1)
            return base_tag in self._tags or base_tag in self._alias_mapping
        
        # 匹配带版本号的标签模式
        version_match = re.match(r'^([A-Z_]+)_V(\d+)$', tag)
        if version_match:
            base_tag = version_match.group(1)
            return base_tag in self._tags or base_tag in self._alias_mapping
        
        return False
    
    def resolve_tag(self, tag: str) -> Optional[str]:
        """
        解析标签，返回标准标签名称
        
        Args:
            tag: 输入标签
            
        Returns:
            Optional[str]: 标准标签名称，如果无效则返回None
        """
        if not self.validate_tag(tag):
            return None
        
        normalized_tag = self._normalize_tag_name(tag)
        
        with self._lock:
            # 直接匹配
            if normalized_tag in self._tags:
                return normalized_tag
            
            # 别名匹配
            if normalized_tag in self._alias_mapping:
                return self._alias_mapping[normalized_tag]
            
            # 模式匹配
            pattern_match = re.match(r'^([A-Z_]+)_(\d+)$', normalized_tag)
            if pattern_match:
                base_tag = pattern_match.group(1)
                if base_tag in self._tags:
                    return normalized_tag  # 返回带数字的完整标签
                elif base_tag in self._alias_mapping:
                    return f"{self._alias_mapping[base_tag]}_{pattern_match.group(2)}"
            
            return None
    
    def format_tag(self, tag: str, level: str = None, with_color: bool = False) -> str:
        """
        格式化标签显示
        
        Args:
            tag: 标签名称
            level: 日志级别
            with_color: 是否包含颜色代码
            
        Returns:
            str: 格式化后的标签
        """
        resolved_tag = self.resolve_tag(tag)
        if not resolved_tag:
            return f"[{tag}]"  # 无效标签的默认格式
        
        # 获取标签信息
        tag_info = self.get_tag_info(resolved_tag)
        
        # 基本格式
        formatted = f"[{resolved_tag}]"
        
        # 添加颜色（如果需要且有颜色信息）
        if with_color and tag_info and tag_info.color_code:
            # 这里可以添加ANSI颜色代码，但为了简单起见暂时跳过
            pass
        
        # 根据级别调整格式
        if level:
            level_upper = level.upper()
            if level_upper in ["ERROR", "CRITICAL"]:
                formatted = f"[!{resolved_tag}!]"
            elif level_upper == "WARNING":
                formatted = f"[*{resolved_tag}*]"
            elif level_upper == "DEBUG":
                formatted = f"[#{resolved_tag}#]"
        
        return formatted
    
    def get_tag_info(self, tag: str) -> Optional[TagInfo]:
        """
        获取标签信息
        
        Args:
            tag: 标签名称
            
        Returns:
            Optional[TagInfo]: 标签信息，如果不存在则返回None
        """
        resolved_tag = self.resolve_tag(tag)
        if not resolved_tag:
            return None
        
        # 处理带数字后缀的标签
        base_tag = re.sub(r'_\d+$', '', resolved_tag)
        
        with self._lock:
            return self._tags.get(base_tag) or self._tags.get(resolved_tag)
    
    def get_tags_by_category(self, category: TagCategory) -> List[str]:
        """
        根据类别获取标签列表
        
        Args:
            category: 标签类别
            
        Returns:
            List[str]: 标签名称列表
        """
        with self._lock:
            return list(self._category_tags.get(category, set()))
    
    def get_tags_by_level(self, level: TagLevel) -> List[str]:
        """
        根据级别获取标签列表
        
        Args:
            level: 标签级别
            
        Returns:
            List[str]: 标签名称列表
        """
        with self._lock:
            return list(self._level_tags.get(level, set()))
    
    def get_child_tags(self, parent_tag: str) -> List[str]:
        """
        获取子标签列表
        
        Args:
            parent_tag: 父标签名称
            
        Returns:
            List[str]: 子标签名称列表
        """
        resolved_parent = self.resolve_tag(parent_tag)
        if not resolved_parent:
            return []
        
        child_tags = []
        with self._lock:
            for tag_name, tag_info in self._tags.items():
                if resolved_parent in tag_info.parent_tags:
                    child_tags.append(tag_name)
        
        return child_tags
    
    def get_parent_tags(self, child_tag: str) -> List[str]:
        """
        获取父标签列表
        
        Args:
            child_tag: 子标签名称
            
        Returns:
            List[str]: 父标签名称列表
        """
        tag_info = self.get_tag_info(child_tag)
        if tag_info:
            return tag_info.parent_tags.copy()
        return []
    
    def record_tag_usage(self, tag: str):
        """
        记录标签使用情况
        
        Args:
            tag: 使用的标签
        """
        resolved_tag = self.resolve_tag(tag)
        if resolved_tag:
            with self._lock:
                self._usage_stats[resolved_tag] += 1
                if resolved_tag in self._tags:
                    self._tags[resolved_tag].usage_count += 1
    
    def get_usage_statistics(self) -> Dict[str, int]:
        """
        获取标签使用统计
        
        Returns:
            Dict[str, int]: 标签使用次数统计
        """
        with self._lock:
            return dict(self._usage_stats)
    
    def get_most_used_tags(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        获取最常用的标签
        
        Args:
            limit: 返回数量限制
            
        Returns:
            List[Tuple[str, int]]: (标签名, 使用次数) 的列表
        """
        with self._lock:
            return self._usage_stats.most_common(limit)
    
    def suggest_tags(self, partial_tag: str, limit: int = 5) -> List[str]:
        """
        根据部分输入建议标签
        
        Args:
            partial_tag: 部分标签输入
            limit: 建议数量限制
            
        Returns:
            List[str]: 建议的标签列表
        """
        if not partial_tag:
            return []
        
        partial_upper = partial_tag.upper()
        suggestions = []
        
        with self._lock:
            # 搜索直接匹配
            for tag_name in self._tags.keys():
                if tag_name.startswith(partial_upper):
                    suggestions.append(tag_name)
            
            # 搜索别名匹配
            for alias, real_tag in self._alias_mapping.items():
                if alias.startswith(partial_upper):
                    suggestions.append(real_tag)
            
            # 搜索包含匹配
            if len(suggestions) < limit:
                for tag_name in self._tags.keys():
                    if partial_upper in tag_name and tag_name not in suggestions:
                        suggestions.append(tag_name)
        
        # 按使用频率排序
        suggestions.sort(key=lambda x: self._usage_stats.get(x, 0), reverse=True)
        
        return suggestions[:limit]
    
    def export_tag_definitions(self) -> Dict[str, Any]:
        """
        导出标签定义
        
        Returns:
            Dict[str, Any]: 标签定义的字典
        """
        with self._lock:
            export_data = {
                'tags': {},
                'aliases': dict(self._alias_mapping),
                'usage_stats': dict(self._usage_stats),
                'categories': {cat.value: list(tags) for cat, tags in self._category_tags.items()},
                'levels': {level.value: list(tags) for level, tags in self._level_tags.items()}
            }
            
            for tag_name, tag_info in self._tags.items():
                export_data['tags'][tag_name] = {
                    'category': tag_info.category.value,
                    'level': tag_info.level.value,
                    'description': tag_info.description,
                    'parent_tags': tag_info.parent_tags,
                    'child_tags': tag_info.child_tags,
                    'aliases': tag_info.aliases,
                    'color_code': tag_info.color_code,
                    'usage_count': tag_info.usage_count
                }
            
            return export_data
    
    def get_tag_hierarchy(self) -> Dict[str, Any]:
        """
        获取标签层次结构
        
        Returns:
            Dict[str, Any]: 标签层次结构
        """
        hierarchy = {}
        
        with self._lock:
            # 获取所有主要标签（没有父标签的标签）
            primary_tags = [name for name, info in self._tags.items() if not info.parent_tags]
            
            def build_hierarchy(tag_name: str) -> Dict[str, Any]:
                tag_info = self._tags.get(tag_name)
                if not tag_info:
                    return {}
                
                node = {
                    'name': tag_name,
                    'category': tag_info.category.value,
                    'level': tag_info.level.value,
                    'description': tag_info.description,
                    'usage_count': tag_info.usage_count,
                    'children': {}
                }
                
                # 递归构建子节点
                child_tags = self.get_child_tags(tag_name)
                for child_tag in child_tags:
                    node['children'][child_tag] = build_hierarchy(child_tag)
                
                return node
            
            for primary_tag in primary_tags:
                hierarchy[primary_tag] = build_hierarchy(primary_tag)
        
        return hierarchy


# 全局标签处理器实例
_global_tag_processor: Optional[StructuredTagProcessor] = None
_processor_lock = threading.Lock()


def get_global_tag_processor() -> StructuredTagProcessor:
    """获取全局标签处理器实例"""
    global _global_tag_processor
    
    with _processor_lock:
        if _global_tag_processor is None:
            _global_tag_processor = StructuredTagProcessor()
        return _global_tag_processor


# 便捷函数
def validate_tag(tag: str) -> bool:
    """验证标签是否有效"""
    processor = get_global_tag_processor()
    return processor.validate_tag(tag)


def format_tag(tag: str, level: str = None) -> str:
    """格式化标签"""
    processor = get_global_tag_processor()
    return processor.format_tag(tag, level)


def suggest_tags(partial_tag: str, limit: int = 5) -> List[str]:
    """建议标签"""
    processor = get_global_tag_processor()
    return processor.suggest_tags(partial_tag, limit)


if __name__ == "__main__":
    # 测试代码
    print("测试结构化标签处理器...")
    
    processor = StructuredTagProcessor()
    
    # 测试标签验证
    test_tags = ["TRAINING", "EPOCH", "BATCH", "INVALID_TAG", "train", "EPOCH_01", "CONFIG"]
    
    print("\n标签验证测试:")
    for tag in test_tags:
        is_valid = processor.validate_tag(tag)
        resolved = processor.resolve_tag(tag)
        formatted = processor.format_tag(tag, "INFO")
        print(f"  {tag:15} -> 有效: {is_valid:5} | 解析: {resolved:15} | 格式化: {formatted}")
    
    # 测试标签建议
    print("\n标签建议测试:")
    partial_inputs = ["TR", "CONF", "OPT", "SYS"]
    for partial in partial_inputs:
        suggestions = processor.suggest_tags(partial)
        print(f"  '{partial}' -> {suggestions}")
    
    # 测试类别查询
    print("\n按类别查询标签:")
    for category in TagCategory:
        tags = processor.get_tags_by_category(category)
        print(f"  {category.value:12} -> {tags}")
    
    # 测试层次结构
    print("\n标签层次结构:")
    hierarchy = processor.get_tag_hierarchy()
    for primary_tag, info in hierarchy.items():
        print(f"  {primary_tag} ({info['category']})")
        for child_tag in info['children']:
            print(f"    └─ {child_tag}")
    
    # 测试使用统计
    print("\n使用统计测试:")
    for tag in ["TRAINING", "CONFIG", "OPTIMIZATION", "ERROR"]:
        processor.record_tag_usage(tag)
        processor.record_tag_usage(tag)  # 记录两次
    
    stats = processor.get_usage_statistics()
    print(f"  使用统计: {stats}")
    
    most_used = processor.get_most_used_tags(5)
    print(f"  最常用标签: {most_used}")
    
    print("\n结构化标签处理器测试完成!")