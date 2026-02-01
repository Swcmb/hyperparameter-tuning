"""
统一日志管理器 (UnifiedLogManager)

负责协调所有日志输出的核心组件，提供统一的接口和格式化功能。
支持多级别、多目标输出，集成Python标准logging模块，提供线程安全的日志处理。

主要功能：
- 带标签的结构化日志输出
- 控制台和文件的双重输出
- 动态配置管理
- 线程安全的并发处理
- 自动文件管理和轮转
"""

import os
import sys
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import queue
import time


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class OutputTarget(Enum):
    """输出目标枚举"""
    CONSOLE = "console"
    FILE = "file"
    BOTH = "both"
    STRUCTURED_FILE = "structured_file"


@dataclass
class LogEntry:
    """日志条目数据结构"""
    timestamp: datetime
    level: str
    tag: str
    component: str
    message: str
    structured_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    thread_id: Optional[int] = None
    
    def to_console_format(self) -> str:
        """转换为控制台显示格式"""
        timestamp_str = self.timestamp.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        component_str = f" - {self.component}" if self.component else ""
        tag_str = f"[{self.tag}]" if self.tag else ""
        
        base_format = f"{timestamp_str}{component_str} - {self.level} - {tag_str} {self.message}"
        
        if self.structured_data:
            # 添加结构化数据的简化显示
            data_str = ", ".join([f"{k}={v}" for k, v in self.structured_data.items() if k not in ['details', 'raw_data']])
            if data_str:
                base_format += f" | {data_str}"
        
        return base_format
    
    def to_file_format(self) -> str:
        """转换为文件存储格式"""
        return self.to_console_format()
    
    def to_json(self) -> str:
        """转换为JSON格式"""
        data = {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'tag': self.tag,
            'component': self.component,
            'message': self.message,
            'thread_id': self.thread_id
        }
        
        if self.structured_data:
            data['structured_data'] = self.structured_data
        if self.context:
            data['context'] = self.context
            
        return json.dumps(data, ensure_ascii=False, separators=(',', ':'), default=str)


@dataclass
class ProgressInfo:
    """进度跟踪数据结构"""
    tag: str
    current: int
    total: int
    percentage: float
    start_time: datetime
    elapsed_time: timedelta
    estimated_remaining: Optional[timedelta] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def format_progress_bar(self, width: int = 50) -> str:
        """格式化进度条显示"""
        filled = int(width * self.percentage / 100)
        bar = '█' * filled + '░' * (width - filled)
        
        elapsed_str = str(self.elapsed_time).split('.')[0]  # 移除微秒
        remaining_str = str(self.estimated_remaining).split('.')[0] if self.estimated_remaining else "未知"
        
        return f"[{bar}] {self.percentage:5.1f}% ({self.current}/{self.total}) | 已用时: {elapsed_str} | 预计剩余: {remaining_str}"


class EmojiRemover:
    """Emoji移除器"""
    
    # 常见emoji到文本的映射
    EMOJI_MAPPING = {
        '🚀': '[启动]',
        '🎯': '[目标]',
        '📊': '[数据]',
        '⚖️': '[权重]',
        '📈': '[进度]',
        '✅': '[完成]',
        '🎉': '[成功]',
        '❌': '[失败]',
        '⚠️': '[警告]',
        '🔧': '[配置]',
        '💾': '[保存]',
        '🔄': '[更新]',
        '📋': '[列表]',
        '🎪': '[测试]',
        '🏆': '[最佳]',
        '📝': '[记录]',
        '🔍': '[搜索]',
        '⏰': '[时间]',
        '💡': '[提示]',
        '🔥': '[热点]',
        '⭐': '[重要]',
        '🎨': '[格式]',
        '🛠️': '[工具]',
        '📦': '[包]',
        '🌟': '[优秀]',
        '🎭': '[模式]',
        '🎲': '[随机]',
        '🎪': '[演示]',
        '🎬': '[开始]',
        '🎤': '[输出]',
        '🎧': '[监听]',
        '🎮': '[控制]',
        '🎯': '[精确]',
        '🎨': '[美化]',
        '🎪': '[展示]'
    }
    
    # Unicode emoji范围的正则表达式
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    
    @classmethod
    def remove_emojis(cls, text: str) -> str:
        """移除文本中的所有emoji"""
        if not text:
            return text
        
        # 首先替换已知的emoji映射
        result = text
        for emoji, replacement in cls.EMOJI_MAPPING.items():
            result = result.replace(emoji, replacement)
        
        # 然后移除剩余的emoji
        result = cls.EMOJI_PATTERN.sub('', result)
        
        # 清理多余的空格
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result


class UnifiedLogManager:
    """
    统一日志管理器
    
    协调所有日志输出的核心组件，提供统一的接口和格式化功能
    """
    
    def __init__(self, 
                 run_name: str,
                 log_level: int = logging.INFO,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 log_dir: str = "logs",
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 max_files: int = 10,
                 buffer_size: int = 1000):
        """
        初始化统一日志管理器
        
        Args:
            run_name: 运行名称，用于文件命名
            log_level: 日志级别
            enable_console: 是否启用控制台输出
            enable_file: 是否启用文件输出
            log_dir: 日志目录
            max_file_size: 最大文件大小（字节）
            max_files: 最大文件数量
            buffer_size: 缓冲区大小
        """
        self.run_name = run_name
        self.log_level = log_level
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.buffer_size = buffer_size
        
        # 创建日志目录
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建运行特定的子目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.log_dir / f"{run_name}_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # 线程安全相关
        self._lock = threading.RLock()
        self._log_queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._worker_thread = None
        
        # 文件处理器
        self._file_handlers = {}
        self._current_file_sizes = {}
        
        # 进度跟踪
        self._progress_trackers = {}
        
        # 组件配置
        self._component_configs = {}
        
        # 初始化日志系统
        self._setup_logging()
        self._start_worker_thread()
        
        # 记录初始化信息
        self.log_with_tag(logging.INFO, "INIT", f"统一日志管理器初始化完成", "UnifiedLogManager")
        self.log_with_tag(logging.INFO, "INIT", f"运行名称: {run_name}", "UnifiedLogManager")
        self.log_with_tag(logging.INFO, "INIT", f"日志目录: {self.run_dir}", "UnifiedLogManager")
        self.log_with_tag(logging.INFO, "INIT", f"控制台输出: {'启用' if enable_console else '禁用'}", "UnifiedLogManager")
        self.log_with_tag(logging.INFO, "INIT", f"文件输出: {'启用' if enable_file else '禁用'}", "UnifiedLogManager")
    
    def _setup_logging(self):
        """设置日志系统"""
        # 创建主日志文件
        if self.enable_file:
            main_log_file = self.run_dir / f"{self.run_name}_main.log"
            structured_log_file = self.run_dir / f"{self.run_name}_structured.jsonl"
            
            # 创建文件处理器
            self._file_handlers['main'] = open(main_log_file, 'w', encoding='utf-8', buffering=1)
            self._file_handlers['structured'] = open(structured_log_file, 'w', encoding='utf-8', buffering=1)
            
            # 初始化文件大小跟踪
            self._current_file_sizes['main'] = 0
            self._current_file_sizes['structured'] = 0
    
    def _start_worker_thread(self):
        """启动工作线程"""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def _worker_loop(self):
        """工作线程主循环"""
        while not self._stop_event.is_set():
            try:
                # 从队列中获取日志条目
                try:
                    log_entry = self._log_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # 处理日志条目
                self._process_log_entry(log_entry)
                self._log_queue.task_done()
                
            except Exception as e:
                # 工作线程中的错误处理
                print(f"[ERROR] 日志工作线程错误: {e}", file=sys.stderr)
    
    def _process_log_entry(self, log_entry: LogEntry):
        """处理单个日志条目"""
        try:
            # 控制台输出
            if self.enable_console:
                console_text = log_entry.to_console_format()
                print(console_text)
            
            # 文件输出
            if self.enable_file and 'main' in self._file_handlers:
                file_text = log_entry.to_file_format()
                self._write_to_file('main', file_text + '\n')
                
                # 结构化文件输出
                if 'structured' in self._file_handlers:
                    json_text = log_entry.to_json()
                    self._write_to_file('structured', json_text + '\n')
        
        except Exception as e:
            print(f"[ERROR] 处理日志条目失败: {e}", file=sys.stderr)
    
    def _write_to_file(self, file_key: str, content: str):
        """写入文件并处理轮转"""
        if file_key not in self._file_handlers:
            return
        
        try:
            handler = self._file_handlers[file_key]
            handler.write(content)
            handler.flush()
            
            # 更新文件大小
            content_size = len(content.encode('utf-8'))
            self._current_file_sizes[file_key] += content_size
            
            # 检查是否需要轮转
            if self._current_file_sizes[file_key] > self.max_file_size:
                self._rotate_file(file_key)
        
        except Exception as e:
            print(f"[ERROR] 写入文件失败 ({file_key}): {e}", file=sys.stderr)
    
    def _rotate_file(self, file_key: str):
        """轮转日志文件"""
        try:
            if file_key not in self._file_handlers:
                return
            
            # 关闭当前文件
            self._file_handlers[file_key].close()
            
            # 生成新的文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if file_key == 'main':
                new_file = self.run_dir / f"{self.run_name}_main_{timestamp}.log"
            elif file_key == 'structured':
                new_file = self.run_dir / f"{self.run_name}_structured_{timestamp}.jsonl"
            else:
                new_file = self.run_dir / f"{self.run_name}_{file_key}_{timestamp}.log"
            
            # 创建新文件处理器
            self._file_handlers[file_key] = open(new_file, 'w', encoding='utf-8', buffering=1)
            self._current_file_sizes[file_key] = 0
            
            print(f"[INFO] 日志文件已轮转: {new_file}")
        
        except Exception as e:
            print(f"[ERROR] 文件轮转失败 ({file_key}): {e}", file=sys.stderr)
    
    def log_with_tag(self, level: int, tag: str, message: str, 
                     component: str = None, **kwargs):
        """
        带标签的日志输出
        
        Args:
            level: 日志级别
            tag: 结构化标签
            message: 日志消息
            component: 组件名称
            **kwargs: 额外的结构化数据
        """
        if level < self.log_level:
            return
        
        # 移除emoji
        clean_message = EmojiRemover.remove_emojis(message)
        clean_tag = EmojiRemover.remove_emojis(tag)
        
        # 创建日志条目
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=logging.getLevelName(level),
            tag=clean_tag,
            component=component or "System",
            message=clean_message,
            structured_data=kwargs if kwargs else None,
            thread_id=threading.get_ident()
        )
        
        # 添加到队列
        try:
            self._log_queue.put_nowait(log_entry)
        except queue.Full:
            # 队列满时直接输出到控制台
            if self.enable_console:
                print(f"[WARNING] 日志队列已满，直接输出: {log_entry.to_console_format()}")
    
    def log_structured(self, level: int, tag: str, data: Dict[str, Any],
                      component: str = None):
        """
        结构化数据日志输出
        
        Args:
            level: 日志级别
            tag: 结构化标签
            data: 结构化数据
            component: 组件名称
        """
        # 生成消息摘要
        message_parts = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if key.endswith('_time') or key.endswith('_duration'):
                    message_parts.append(f"{key}={value:.4f}s")
                elif isinstance(value, float) and abs(value) < 0.001:
                    message_parts.append(f"{key}={value:.2e}")
                else:
                    message_parts.append(f"{key}={value}")
            elif isinstance(value, str) and len(value) > 50:
                message_parts.append(f"{key}={value[:47]}...")
            else:
                message_parts.append(f"{key}={value}")
        
        message = " | ".join(message_parts)
        
        self.log_with_tag(level, tag, message, component, **data)
    
    def log_progress(self, tag: str, current: int, total: int, 
                    message: str = "", **metrics):
        """
        进度日志输出
        
        Args:
            tag: 进度标签
            current: 当前进度
            total: 总进度
            message: 附加消息
            **metrics: 进度指标
        """
        # 计算进度信息
        percentage = (current / total * 100) if total > 0 else 0
        
        # 获取或创建进度跟踪器
        if tag not in self._progress_trackers:
            self._progress_trackers[tag] = {
                'start_time': datetime.now(),
                'last_update': datetime.now(),
                'last_current': 0
            }
        
        tracker = self._progress_trackers[tag]
        now = datetime.now()
        elapsed = now - tracker['start_time']
        
        # 估算剩余时间
        estimated_remaining = None
        if current > tracker['last_current'] and current > 0:
            rate = (current - tracker['last_current']) / (now - tracker['last_update']).total_seconds()
            if rate > 0:
                remaining_items = total - current
                estimated_remaining = timedelta(seconds=remaining_items / rate)
        
        # 更新跟踪器
        tracker['last_update'] = now
        tracker['last_current'] = current
        
        # 创建进度信息
        progress_info = ProgressInfo(
            tag=tag,
            current=current,
            total=total,
            percentage=percentage,
            start_time=tracker['start_time'],
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining,
            metrics=metrics
        )
        
        # 格式化进度消息
        progress_bar = progress_info.format_progress_bar()
        full_message = f"{message} {progress_bar}" if message else progress_bar
        
        # 添加指标信息
        if metrics:
            metrics_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
            full_message += f" | {metrics_str}"
        
        self.log_with_tag(logging.INFO, "PROGRESS", full_message, tag)
    
    def set_component_config(self, component: str, config: Dict[str, Any]):
        """
        设置组件配置
        
        Args:
            component: 组件名称
            config: 配置字典
        """
        with self._lock:
            self._component_configs[component] = config
            
        self.log_with_tag(logging.INFO, "CONFIG", 
                         f"组件配置已更新: {component}", 
                         component, **config)
    
    def get_component_config(self, component: str) -> Dict[str, Any]:
        """获取组件配置"""
        with self._lock:
            return self._component_configs.get(component, {})
    
    def flush(self):
        """刷新所有缓冲区"""
        # 等待队列清空
        self._log_queue.join()
        
        # 刷新文件处理器
        for handler in self._file_handlers.values():
            if hasattr(handler, 'flush'):
                handler.flush()
    
    def close(self):
        """关闭日志管理器"""
        self.log_with_tag(logging.INFO, "SHUTDOWN", "正在关闭统一日志管理器", "UnifiedLogManager")
        
        # 停止工作线程
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        
        # 刷新并关闭文件处理器
        for handler in self._file_handlers.values():
            try:
                handler.flush()
                handler.close()
            except:
                pass
        
        self._file_handlers.clear()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 全局日志管理器实例
_global_log_manager: Optional[UnifiedLogManager] = None
_global_lock = threading.Lock()


def get_global_log_manager() -> Optional[UnifiedLogManager]:
    """获取全局日志管理器实例"""
    return _global_log_manager


def init_global_log_manager(run_name: str, **kwargs) -> UnifiedLogManager:
    """
    初始化全局日志管理器
    
    Args:
        run_name: 运行名称
        **kwargs: 其他初始化参数
        
    Returns:
        UnifiedLogManager实例
    """
    global _global_log_manager
    
    with _global_lock:
        if _global_log_manager is not None:
            _global_log_manager.close()
        
        _global_log_manager = UnifiedLogManager(run_name, **kwargs)
        return _global_log_manager


def close_global_log_manager():
    """关闭全局日志管理器"""
    global _global_log_manager
    
    with _global_lock:
        if _global_log_manager is not None:
            _global_log_manager.close()
            _global_log_manager = None


# 便捷函数
def log_info(tag: str, message: str, component: str = None, **kwargs):
    """便捷的INFO级别日志函数"""
    manager = get_global_log_manager()
    if manager:
        manager.log_with_tag(logging.INFO, tag, message, component, **kwargs)


def log_warning(tag: str, message: str, component: str = None, **kwargs):
    """便捷的WARNING级别日志函数"""
    manager = get_global_log_manager()
    if manager:
        manager.log_with_tag(logging.WARNING, tag, message, component, **kwargs)


def log_error(tag: str, message: str, component: str = None, **kwargs):
    """便捷的ERROR级别日志函数"""
    manager = get_global_log_manager()
    if manager:
        manager.log_with_tag(logging.ERROR, tag, message, component, **kwargs)


def log_debug(tag: str, message: str, component: str = None, **kwargs):
    """便捷的DEBUG级别日志函数"""
    manager = get_global_log_manager()
    if manager:
        manager.log_with_tag(logging.DEBUG, tag, message, component, **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("测试统一日志管理器...")
    
    # 创建测试实例
    with UnifiedLogManager("test_run", log_level=logging.DEBUG) as manager:
        # 测试基本日志输出
        manager.log_with_tag(logging.INFO, "TEST", "这是一个测试消息", "TestComponent")
        
        # 测试emoji移除
        manager.log_with_tag(logging.INFO, "TEST", "🚀 开始测试 📊 数据处理", "TestComponent")
        
        # 测试结构化日志
        manager.log_structured(logging.INFO, "METRICS", {
            "accuracy": 0.95,
            "loss": 0.05,
            "epoch": 10,
            "batch_size": 32
        }, "TestComponent")
        
        # 测试进度日志
        for i in range(0, 101, 10):
            manager.log_progress("TRAINING", i, 100, "训练进度", 
                               loss=1.0 - i/100, accuracy=i/100)
            time.sleep(0.1)
        
        # 测试组件配置
        manager.set_component_config("TestComponent", {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        })
        
        print("测试完成，检查日志文件...")
    
    print("统一日志管理器测试完成!")