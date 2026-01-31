"""
性能基准测试 (Performance Benchmark)

用于测试输出系统在各种场景下的性能表现，包括：
- 大量日志输出的性能测试
- 并发场景下的性能表现
- 文件I/O操作的效率验证
- 内存使用情况监控
- 长期运行稳定性测试
"""

import os
import sys
import time
import threading
import multiprocessing
import psutil
import gc
import json
import random
import string
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
import logging

# 导入输出系统组件
from unified_log_manager import UnifiedLogManager, init_global_log_manager, close_global_log_manager
from structured_tag_processor import StructuredTagProcessor
from output_formatter import OutputFormatter
from file_manager import FileManager


@dataclass
class BenchmarkResult:
    """基准测试结果数据结构"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    operations_count: int
    operations_per_second: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    success_rate: float
    error_count: int
    additional_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        return result


@dataclass
class SystemMetrics:
    """系统指标数据结构"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    thread_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, interval: float = 0.1):
        """
        初始化系统监控器
        
        Args:
            interval: 监控间隔（秒）
        """
        self.interval = interval
        self.monitoring = False
        self.metrics: List[SystemMetrics] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
        
        # 初始化磁盘I/O计数器
        self.initial_io = psutil.disk_io_counters()
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 获取系统指标
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                
                # 获取磁盘I/O
                current_io = psutil.disk_io_counters()
                if current_io and self.initial_io:
                    disk_read_mb = (current_io.read_bytes - self.initial_io.read_bytes) / (1024 * 1024)
                    disk_write_mb = (current_io.write_bytes - self.initial_io.write_bytes) / (1024 * 1024)
                else:
                    disk_read_mb = disk_write_mb = 0.0
                
                # 创建指标记录
                metrics = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_used_mb=memory_info.rss / (1024 * 1024),
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    thread_count=threading.active_count()
                )
                
                self.metrics.append(metrics)
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"[ERROR] 监控错误: {e}")
                time.sleep(self.interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in self.metrics]
        memory_values = [m.memory_used_mb for m in self.metrics]
        
        return {
            'duration_seconds': (self.metrics[-1].timestamp - self.metrics[0].timestamp).total_seconds(),
            'sample_count': len(self.metrics),
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_mb': sum(memory_values) / len(memory_values),
            'peak_memory_mb': max(memory_values),
            'final_disk_read_mb': self.metrics[-1].disk_io_read_mb,
            'final_disk_write_mb': self.metrics[-1].disk_io_write_mb,
            'max_thread_count': max(m.thread_count for m in self.metrics)
        }


class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        初始化性能基准测试器
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
        self.system_monitor = SystemMonitor()
        
        # 设置日志
        self.logger = logging.getLogger("PerformanceBenchmark")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def run_all_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """运行所有基准测试"""
        self.logger.info("[BENCHMARK] 开始运行所有性能基准测试")
        
        benchmark_methods = [
            self.benchmark_basic_logging,
            self.benchmark_high_volume_logging,
            self.benchmark_concurrent_logging,
            self.benchmark_structured_data_logging,
            self.benchmark_file_operations,
            self.benchmark_memory_usage,
            self.benchmark_tag_processing,
            self.benchmark_formatting_performance,
            self.benchmark_long_running_stability
        ]
        
        results = {}
        
        for benchmark_method in benchmark_methods:
            try:
                self.logger.info(f"[BENCHMARK] 运行测试: {benchmark_method.__name__}")
                result = benchmark_method()
                results[result.test_name] = result
                self.results.append(result)
                
                # 清理内存
                gc.collect()
                time.sleep(1.0)  # 短暂休息
                
            except Exception as e:
                self.logger.error(f"[ERROR] 测试失败 {benchmark_method.__name__}: {e}")
                continue
        
        # 生成综合报告
        self._generate_comprehensive_report(results)
        
        self.logger.info(f"[BENCHMARK] 所有测试完成，共运行 {len(results)} 个测试")
        return results
    
    def benchmark_basic_logging(self) -> BenchmarkResult:
        """基础日志记录性能测试"""
        test_name = "basic_logging"
        operations_count = 10000
        
        self.logger.info(f"[BENCHMARK] 开始基础日志测试 - {operations_count} 条日志")
        
        # 开始监控
        self.system_monitor.start_monitoring()
        start_time = datetime.now()
        
        # 创建日志管理器
        with UnifiedLogManager(f"benchmark_{test_name}", enable_console=False) as log_manager:
            error_count = 0
            
            for i in range(operations_count):
                try:
                    log_manager.log_with_tag(
                        logging.INFO, 
                        "BENCHMARK", 
                        f"基础日志测试消息 {i:05d}",
                        "BenchmarkTest",
                        iteration=i,
                        test_type="basic"
                    )
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # 只记录前5个错误
                        self.logger.error(f"[ERROR] 日志写入失败 {i}: {e}")
        
        end_time = datetime.now()
        self.system_monitor.stop_monitoring()
        
        # 计算结果
        duration = (end_time - start_time).total_seconds()
        ops_per_second = operations_count / duration if duration > 0 else 0
        success_rate = (operations_count - error_count) / operations_count * 100
        
        monitor_summary = self.system_monitor.get_summary()
        
        return BenchmarkResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            operations_count=operations_count,
            operations_per_second=ops_per_second,
            memory_usage_mb=monitor_summary.get('avg_memory_mb', 0),
            peak_memory_mb=monitor_summary.get('peak_memory_mb', 0),
            cpu_usage_percent=monitor_summary.get('avg_cpu_percent', 0),
            disk_io_read_mb=monitor_summary.get('final_disk_read_mb', 0),
            disk_io_write_mb=monitor_summary.get('final_disk_write_mb', 0),
            success_rate=success_rate,
            error_count=error_count,
            additional_metrics={
                'max_cpu_percent': monitor_summary.get('max_cpu_percent', 0),
                'max_thread_count': monitor_summary.get('max_thread_count', 0)
            }
        )
    
    def benchmark_high_volume_logging(self) -> BenchmarkResult:
        """大量日志输出性能测试"""
        test_name = "high_volume_logging"
        operations_count = 100000  # 10万条日志
        
        self.logger.info(f"[BENCHMARK] 开始大量日志测试 - {operations_count} 条日志")
        
        self.system_monitor.start_monitoring()
        start_time = datetime.now()
        
        with UnifiedLogManager(f"benchmark_{test_name}", 
                             enable_console=False,
                             buffer_size=5000) as log_manager:
            error_count = 0
            
            # 生成随机测试数据
            test_messages = [
                f"高容量日志测试消息 {i:06d} - {''.join(random.choices(string.ascii_letters, k=50))}"
                for i in range(min(1000, operations_count // 10))  # 预生成部分消息以减少字符串生成开销
            ]
            
            for i in range(operations_count):
                try:
                    message = test_messages[i % len(test_messages)]
                    log_manager.log_with_tag(
                        logging.INFO,
                        "HIGH_VOLUME",
                        message,
                        "BenchmarkTest",
                        batch_id=i // 1000,
                        sequence=i,
                        random_value=random.random()
                    )
                    
                    # 每1000条记录一次进度
                    if i % 1000 == 0 and i > 0:
                        log_manager.log_progress("BENCHMARK", i, operations_count, "大量日志测试进度")
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        self.logger.error(f"[ERROR] 大量日志写入失败 {i}: {e}")
        
        end_time = datetime.now()
        self.system_monitor.stop_monitoring()
        
        duration = (end_time - start_time).total_seconds()
        ops_per_second = operations_count / duration if duration > 0 else 0
        success_rate = (operations_count - error_count) / operations_count * 100
        
        monitor_summary = self.system_monitor.get_summary()
        
        return BenchmarkResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            operations_count=operations_count,
            operations_per_second=ops_per_second,
            memory_usage_mb=monitor_summary.get('avg_memory_mb', 0),
            peak_memory_mb=monitor_summary.get('peak_memory_mb', 0),
            cpu_usage_percent=monitor_summary.get('avg_cpu_percent', 0),
            disk_io_read_mb=monitor_summary.get('final_disk_read_mb', 0),
            disk_io_write_mb=monitor_summary.get('final_disk_write_mb', 0),
            success_rate=success_rate,
            error_count=error_count,
            additional_metrics={
                'throughput_mb_per_second': (monitor_summary.get('final_disk_write_mb', 0) / duration) if duration > 0 else 0,
                'avg_message_length': 50 + 30  # 大致的消息长度
            }
        )
    
    def benchmark_concurrent_logging(self) -> BenchmarkResult:
        """并发日志记录性能测试"""
        test_name = "concurrent_logging"
        thread_count = 10
        operations_per_thread = 5000
        total_operations = thread_count * operations_per_thread
        
        self.logger.info(f"[BENCHMARK] 开始并发日志测试 - {thread_count} 线程，每线程 {operations_per_thread} 条日志")
        
        self.system_monitor.start_monitoring()
        start_time = datetime.now()
        
        # 共享错误计数器
        total_error_count = [0]  # 使用列表来避免threading.local问题
        error_lock = threading.Lock()
        
        def worker_thread(thread_id: int, log_manager: UnifiedLogManager):
            """工作线程函数"""
            local_errors = 0
            
            for i in range(operations_per_thread):
                try:
                    log_manager.log_with_tag(
                        logging.INFO,
                        "CONCURRENT",
                        f"并发测试 - 线程 {thread_id:02d} 消息 {i:04d}",
                        f"Thread-{thread_id}",
                        thread_id=thread_id,
                        message_id=i,
                        timestamp=time.time()
                    )
                except Exception as e:
                    local_errors += 1
                    if local_errors <= 3:  # 每个线程最多记录3个错误
                        self.logger.error(f"[ERROR] 并发日志失败 T{thread_id}-{i}: {e}")
            
            # 更新全局错误计数
            with error_lock:
                total_error_count[0] += local_errors
        
        with UnifiedLogManager(f"benchmark_{test_name}", 
                             enable_console=False,
                             buffer_size=10000) as log_manager:
            
            # 创建并启动线程
            threads = []
            for thread_id in range(thread_count):
                thread = threading.Thread(
                    target=worker_thread,
                    args=(thread_id, log_manager),
                    name=f"BenchmarkWorker-{thread_id}"
                )
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
        
        end_time = datetime.now()
        self.system_monitor.stop_monitoring()
        
        duration = (end_time - start_time).total_seconds()
        ops_per_second = total_operations / duration if duration > 0 else 0
        success_rate = (total_operations - total_error_count[0]) / total_operations * 100
        
        monitor_summary = self.system_monitor.get_summary()
        
        return BenchmarkResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            operations_count=total_operations,
            operations_per_second=ops_per_second,
            memory_usage_mb=monitor_summary.get('avg_memory_mb', 0),
            peak_memory_mb=monitor_summary.get('peak_memory_mb', 0),
            cpu_usage_percent=monitor_summary.get('avg_cpu_percent', 0),
            disk_io_read_mb=monitor_summary.get('final_disk_read_mb', 0),
            disk_io_write_mb=monitor_summary.get('final_disk_write_mb', 0),
            success_rate=success_rate,
            error_count=total_error_count[0],
            additional_metrics={
                'thread_count': thread_count,
                'operations_per_thread': operations_per_thread,
                'max_thread_count': monitor_summary.get('max_thread_count', 0)
            }
        )
    
    def benchmark_structured_data_logging(self) -> BenchmarkResult:
        """结构化数据日志性能测试"""
        test_name = "structured_data_logging"
        operations_count = 20000
        
        self.logger.info(f"[BENCHMARK] 开始结构化数据日志测试 - {operations_count} 条记录")
        
        self.system_monitor.start_monitoring()
        start_time = datetime.now()
        
        with UnifiedLogManager(f"benchmark_{test_name}", enable_console=False) as log_manager:
            error_count = 0
            
            for i in range(operations_count):
                try:
                    # 生成复杂的结构化数据
                    structured_data = {
                        'iteration': i,
                        'timestamp': time.time(),
                        'metrics': {
                            'accuracy': random.uniform(0.7, 0.95),
                            'loss': random.uniform(0.01, 0.5),
                            'f1_score': random.uniform(0.6, 0.9),
                            'precision': random.uniform(0.65, 0.92),
                            'recall': random.uniform(0.68, 0.88)
                        },
                        'hyperparameters': {
                            'learning_rate': random.uniform(0.0001, 0.01),
                            'batch_size': random.choice([16, 32, 64, 128]),
                            'hidden_layers': random.randint(2, 5),
                            'dropout_rate': random.uniform(0.1, 0.5)
                        },
                        'system_info': {
                            'gpu_memory': random.uniform(1.0, 8.0),
                            'cpu_usage': random.uniform(10.0, 80.0),
                            'disk_io': random.uniform(0.1, 10.0)
                        },
                        'tags': [f"tag_{j}" for j in range(random.randint(1, 5))],
                        'description': f"结构化数据测试记录 {i:05d} - " + ''.join(random.choices(string.ascii_letters, k=30))
                    }
                    
                    log_manager.log_structured(
                        logging.INFO,
                        "STRUCTURED",
                        structured_data,
                        "BenchmarkTest"
                    )
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        self.logger.error(f"[ERROR] 结构化日志失败 {i}: {e}")
        
        end_time = datetime.now()
        self.system_monitor.stop_monitoring()
        
        duration = (end_time - start_time).total_seconds()
        ops_per_second = operations_count / duration if duration > 0 else 0
        success_rate = (operations_count - error_count) / operations_count * 100
        
        monitor_summary = self.system_monitor.get_summary()
        
        return BenchmarkResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            operations_count=operations_count,
            operations_per_second=ops_per_second,
            memory_usage_mb=monitor_summary.get('avg_memory_mb', 0),
            peak_memory_mb=monitor_summary.get('peak_memory_mb', 0),
            cpu_usage_percent=monitor_summary.get('avg_cpu_percent', 0),
            disk_io_read_mb=monitor_summary.get('final_disk_read_mb', 0),
            disk_io_write_mb=monitor_summary.get('final_disk_write_mb', 0),
            success_rate=success_rate,
            error_count=error_count,
            additional_metrics={
                'avg_data_complexity': 15,  # 大致的数据字段数量
                'json_serialization_overhead': True
            }
        )
    
    def benchmark_file_operations(self) -> BenchmarkResult:
        """文件操作性能测试"""
        test_name = "file_operations"
        operations_count = 5000
        
        self.logger.info(f"[BENCHMARK] 开始文件操作测试 - {operations_count} 次操作")
        
        self.system_monitor.start_monitoring()
        start_time = datetime.now()
        
        error_count = 0
        
        with FileManager(f"benchmark_{test_name}_files", f"benchmark_{test_name}") as file_manager:
            # 创建日志文件
            created_files = file_manager.create_log_files()
            
            for i in range(operations_count):
                try:
                    # 轮换不同类型的文件操作
                    operation_type = i % 4
                    
                    if operation_type == 0:
                        # 写入主日志
                        success = file_manager.write_to_file(
                            'main',
                            f"文件操作测试 {i:05d} - 主日志记录",
                            flush=(i % 100 == 0)  # 每100次刷新一次
                        )
                        if not success:
                            error_count += 1
                    
                    elif operation_type == 1:
                        # 写入训练日志
                        success = file_manager.write_to_file(
                            'training',
                            f"训练日志 {i:05d} - Epoch {i//100}, Batch {i%100}",
                            flush=(i % 50 == 0)
                        )
                        if not success:
                            error_count += 1
                    
                    elif operation_type == 2:
                        # 写入结构化数据
                        data = {
                            'operation_id': i,
                            'type': 'file_benchmark',
                            'value': random.random(),
                            'timestamp': time.time()
                        }
                        success = file_manager.write_structured_data('metrics', data)
                        if not success:
                            error_count += 1
                    
                    else:
                        # 写入系统日志
                        success = file_manager.write_to_file(
                            'system',
                            f"系统日志 {i:05d} - CPU: {random.uniform(10, 80):.1f}%, Memory: {random.uniform(1, 8):.2f}GB"
                        )
                        if not success:
                            error_count += 1
                    
                    # 定期检查磁盘空间
                    if i % 1000 == 0:
                        file_manager.monitor_disk_space()
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        self.logger.error(f"[ERROR] 文件操作失败 {i}: {e}")
        
        end_time = datetime.now()
        self.system_monitor.stop_monitoring()
        
        duration = (end_time - start_time).total_seconds()
        ops_per_second = operations_count / duration if duration > 0 else 0
        success_rate = (operations_count - error_count) / operations_count * 100
        
        monitor_summary = self.system_monitor.get_summary()
        
        return BenchmarkResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            operations_count=operations_count,
            operations_per_second=ops_per_second,
            memory_usage_mb=monitor_summary.get('avg_memory_mb', 0),
            peak_memory_mb=monitor_summary.get('peak_memory_mb', 0),
            cpu_usage_percent=monitor_summary.get('avg_cpu_percent', 0),
            disk_io_read_mb=monitor_summary.get('final_disk_read_mb', 0),
            disk_io_write_mb=monitor_summary.get('final_disk_write_mb', 0),
            success_rate=success_rate,
            error_count=error_count,
            additional_metrics={
                'file_types_tested': 4,
                'disk_space_checks': operations_count // 1000
            }
        )
    
    def benchmark_memory_usage(self) -> BenchmarkResult:
        """内存使用性能测试"""
        test_name = "memory_usage"
        operations_count = 50000
        
        self.logger.info(f"[BENCHMARK] 开始内存使用测试 - {operations_count} 次操作")
        
        # 记录初始内存使用
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        self.system_monitor.start_monitoring()
        start_time = datetime.now()
        
        # 创建大量日志管理器实例来测试内存管理
        log_managers = []
        error_count = 0
        
        try:
            for i in range(operations_count):
                try:
                    if i % 1000 == 0:
                        # 每1000次创建一个新的日志管理器
                        log_manager = UnifiedLogManager(
                            f"memory_test_{i//1000}",
                            enable_console=False,
                            enable_file=True
                        )
                        log_managers.append(log_manager)
                        
                        # 限制同时存在的日志管理器数量
                        if len(log_managers) > 10:
                            old_manager = log_managers.pop(0)
                            old_manager.close()
                    
                    # 使用当前的日志管理器
                    if log_managers:
                        current_manager = log_managers[-1]
                        current_manager.log_with_tag(
                            logging.INFO,
                            "MEMORY_TEST",
                            f"内存测试消息 {i:05d} - {'x' * (i % 100)}",  # 变长消息
                            "MemoryBenchmark",
                            iteration=i,
                            memory_mb=psutil.Process().memory_info().rss / (1024 * 1024)
                        )
                    
                    # 定期强制垃圾回收
                    if i % 5000 == 0:
                        gc.collect()
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        self.logger.error(f"[ERROR] 内存测试失败 {i}: {e}")
        
        finally:
            # 清理所有日志管理器
            for manager in log_managers:
                try:
                    manager.close()
                except:
                    pass
        
        end_time = datetime.now()
        self.system_monitor.stop_monitoring()
        
        # 最终内存使用
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_growth = final_memory - initial_memory
        
        duration = (end_time - start_time).total_seconds()
        ops_per_second = operations_count / duration if duration > 0 else 0
        success_rate = (operations_count - error_count) / operations_count * 100
        
        monitor_summary = self.system_monitor.get_summary()
        
        return BenchmarkResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            operations_count=operations_count,
            operations_per_second=ops_per_second,
            memory_usage_mb=monitor_summary.get('avg_memory_mb', 0),
            peak_memory_mb=monitor_summary.get('peak_memory_mb', 0),
            cpu_usage_percent=monitor_summary.get('avg_cpu_percent', 0),
            disk_io_read_mb=monitor_summary.get('final_disk_read_mb', 0),
            disk_io_write_mb=monitor_summary.get('final_disk_write_mb', 0),
            success_rate=success_rate,
            error_count=error_count,
            additional_metrics={
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_growth_mb': memory_growth,
                'log_managers_created': operations_count // 1000,
                'gc_collections': operations_count // 5000
            }
        )
    
    def benchmark_tag_processing(self) -> BenchmarkResult:
        """标签处理性能测试"""
        test_name = "tag_processing"
        operations_count = 100000
        
        self.logger.info(f"[BENCHMARK] 开始标签处理测试 - {operations_count} 次操作")
        
        self.system_monitor.start_monitoring()
        start_time = datetime.now()
        
        processor = StructuredTagProcessor()
        error_count = 0
        
        # 测试标签列表
        test_tags = [
            "TRAINING", "EPOCH", "BATCH", "LOSS", "METRICS",
            "CONFIG", "PARAMS", "SETUP", "INIT",
            "OPTIMIZATION", "ACQUISITION", "SUGGESTION",
            "SYSTEM", "MEMORY", "GPU", "CPU",
            "ERROR", "WARNING", "INFO", "DEBUG",
            "RESULTS", "ANALYSIS", "SUMMARY",
            "PROGRESS", "CHECKPOINT", "MILESTONE",
            "DATA", "DATA_STATS", "DATA_LOAD",
            "MODEL", "MODEL_INIT", "MODEL_SAVE",
            "EVALUATION", "TEST", "BENCHMARK",
            # 一些无效标签用于测试
            "INVALID_TAG", "unknown", "test_123", "EPOCH_01"
        ]
        
        for i in range(operations_count):
            try:
                tag = test_tags[i % len(test_tags)]
                
                # 执行各种标签操作
                is_valid = processor.validate_tag(tag)
                resolved = processor.resolve_tag(tag)
                formatted = processor.format_tag(tag, "INFO")
                
                if is_valid:
                    processor.record_tag_usage(tag)
                    tag_info = processor.get_tag_info(tag)
                
                # 每1000次执行一些复杂操作
                if i % 1000 == 0:
                    suggestions = processor.suggest_tags(tag[:3])
                    hierarchy = processor.get_tag_hierarchy()
                    stats = processor.get_usage_statistics()
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    self.logger.error(f"[ERROR] 标签处理失败 {i}: {e}")
        
        end_time = datetime.now()
        self.system_monitor.stop_monitoring()
        
        duration = (end_time - start_time).total_seconds()
        ops_per_second = operations_count / duration if duration > 0 else 0
        success_rate = (operations_count - error_count) / operations_count * 100
        
        monitor_summary = self.system_monitor.get_summary()
        
        # 获取标签使用统计
        final_stats = processor.get_usage_statistics()
        most_used = processor.get_most_used_tags(10)
        
        return BenchmarkResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            operations_count=operations_count,
            operations_per_second=ops_per_second,
            memory_usage_mb=monitor_summary.get('avg_memory_mb', 0),
            peak_memory_mb=monitor_summary.get('peak_memory_mb', 0),
            cpu_usage_percent=monitor_summary.get('avg_cpu_percent', 0),
            disk_io_read_mb=monitor_summary.get('final_disk_read_mb', 0),
            disk_io_write_mb=monitor_summary.get('final_disk_write_mb', 0),
            success_rate=success_rate,
            error_count=error_count,
            additional_metrics={
                'unique_tags_tested': len(test_tags),
                'total_tag_usages': sum(final_stats.values()),
                'most_used_tag': most_used[0] if most_used else None,
                'hierarchy_complexity': len(processor.get_tag_hierarchy())
            }
        )
    
    def benchmark_formatting_performance(self) -> BenchmarkResult:
        """格式化性能测试"""
        test_name = "formatting_performance"
        operations_count = 30000
        
        self.logger.info(f"[BENCHMARK] 开始格式化性能测试 - {operations_count} 次操作")
        
        self.system_monitor.start_monitoring()
        start_time = datetime.now()
        
        formatter = OutputFormatter()
        error_count = 0
        
        for i in range(operations_count):
            try:
                operation_type = i % 6
                
                if operation_type == 0:
                    # 训练信息格式化
                    lines = formatter.format_training_info(
                        epoch=i % 50 + 1,
                        total_epochs=50,
                        batch=i % 100,
                        total_batches=100,
                        loss_info={
                            'total': random.uniform(0.1, 1.0),
                            'bce': random.uniform(0.05, 0.5),
                            'contrast': random.uniform(0.01, 0.3)
                        },
                        metrics={
                            'auroc': random.uniform(0.7, 0.95),
                            'f1': random.uniform(0.6, 0.9)
                        }
                    )
                
                elif operation_type == 1:
                    # 优化信息格式化
                    lines = formatter.format_optimization_info(
                        iteration=i,
                        suggested_params={
                            'lr': random.uniform(0.0001, 0.01),
                            'batch_size': random.choice([16, 32, 64]),
                            'hidden_dim': random.choice([64, 128, 256])
                        },
                        acquisition_value=random.uniform(0.1, 1.0)
                    )
                
                elif operation_type == 2:
                    # 系统信息格式化
                    lines = formatter.format_system_info(
                        gpu_info={
                            'name': 'Test GPU',
                            'total_memory': 8.0,
                            'allocated_memory': random.uniform(1.0, 6.0),
                            'utilization': random.uniform(10.0, 90.0)
                        },
                        memory_info={
                            'total': 16.0,
                            'used': random.uniform(4.0, 12.0),
                            'percent': random.uniform(25.0, 75.0)
                        }
                    )
                
                elif operation_type == 3:
                    # 错误信息格式化
                    test_error = ValueError(f"测试错误 {i}")
                    lines = formatter.format_error_info(
                        test_error,
                        context={'iteration': i, 'test_type': 'benchmark'},
                        component='BenchmarkTest'
                    )
                
                elif operation_type == 4:
                    # 进度条格式化
                    progress_bar = formatter.format_progress_bar(
                        current=i % 1000,
                        total=1000,
                        prefix="测试进度"
                    )
                
                else:
                    # Emoji移除测试
                    test_text = f"🚀 测试消息 {i} ✅ 完成 📊 统计 ⚠️ 警告"
                    cleaned_text = formatter.remove_emojis(test_text)
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    self.logger.error(f"[ERROR] 格式化失败 {i}: {e}")
        
        end_time = datetime.now()
        self.system_monitor.stop_monitoring()
        
        duration = (end_time - start_time).total_seconds()
        ops_per_second = operations_count / duration if duration > 0 else 0
        success_rate = (operations_count - error_count) / operations_count * 100
        
        monitor_summary = self.system_monitor.get_summary()
        
        return BenchmarkResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            operations_count=operations_count,
            operations_per_second=ops_per_second,
            memory_usage_mb=monitor_summary.get('avg_memory_mb', 0),
            peak_memory_mb=monitor_summary.get('peak_memory_mb', 0),
            cpu_usage_percent=monitor_summary.get('avg_cpu_percent', 0),
            disk_io_read_mb=monitor_summary.get('final_disk_read_mb', 0),
            disk_io_write_mb=monitor_summary.get('final_disk_write_mb', 0),
            success_rate=success_rate,
            error_count=error_count,
            additional_metrics={
                'formatting_types_tested': 6,
                'emoji_removal_tests': operations_count // 6
            }
        )
    
    def benchmark_long_running_stability(self) -> BenchmarkResult:
        """长期运行稳定性测试"""
        test_name = "long_running_stability"
        duration_minutes = 5  # 5分钟的稳定性测试
        operations_per_second = 100  # 每秒100次操作
        
        self.logger.info(f"[BENCHMARK] 开始长期运行稳定性测试 - {duration_minutes} 分钟")
        
        self.system_monitor.start_monitoring()
        start_time = datetime.now()
        end_target = start_time + timedelta(minutes=duration_minutes)
        
        operations_count = 0
        error_count = 0
        
        with UnifiedLogManager(f"benchmark_{test_name}", 
                             enable_console=False,
                             buffer_size=2000) as log_manager:
            
            while datetime.now() < end_target:
                try:
                    # 模拟各种类型的日志操作
                    operation_type = operations_count % 5
                    
                    if operation_type == 0:
                        log_manager.log_with_tag(
                            logging.INFO,
                            "STABILITY",
                            f"稳定性测试消息 {operations_count:06d}",
                            "StabilityTest",
                            operation_count=operations_count,
                            elapsed_minutes=(datetime.now() - start_time).total_seconds() / 60
                        )
                    
                    elif operation_type == 1:
                        log_manager.log_structured(
                            logging.INFO,
                            "METRICS",
                            {
                                'operation_id': operations_count,
                                'cpu_percent': psutil.cpu_percent(),
                                'memory_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                                'timestamp': time.time()
                            },
                            "StabilityTest"
                        )
                    
                    elif operation_type == 2:
                        log_manager.log_progress(
                            "STABILITY_PROGRESS",
                            operations_count % 1000,
                            1000,
                            "长期运行进度",
                            rate=operations_per_second
                        )
                    
                    elif operation_type == 3:
                        # 模拟警告日志
                        log_manager.log_with_tag(
                            logging.WARNING,
                            "WARNING",
                            f"模拟警告 {operations_count} - 内存使用较高",
                            "StabilityTest"
                        )
                    
                    else:
                        # 模拟错误日志
                        log_manager.log_with_tag(
                            logging.ERROR,
                            "ERROR",
                            f"模拟错误 {operations_count} - 连接超时",
                            "StabilityTest"
                        )
                    
                    operations_count += 1
                    
                    # 控制操作频率
                    time.sleep(1.0 / operations_per_second)
                    
                    # 定期内存清理
                    if operations_count % 1000 == 0:
                        gc.collect()
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:
                        self.logger.error(f"[ERROR] 稳定性测试失败 {operations_count}: {e}")
                    
                    # 如果错误太多，提前结束
                    if error_count > 100:
                        self.logger.error("[ERROR] 错误过多，提前结束稳定性测试")
                        break
        
        end_time = datetime.now()
        self.system_monitor.stop_monitoring()
        
        actual_duration = (end_time - start_time).total_seconds()
        ops_per_second = operations_count / actual_duration if actual_duration > 0 else 0
        success_rate = (operations_count - error_count) / operations_count * 100 if operations_count > 0 else 0
        
        monitor_summary = self.system_monitor.get_summary()
        
        return BenchmarkResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=actual_duration,
            operations_count=operations_count,
            operations_per_second=ops_per_second,
            memory_usage_mb=monitor_summary.get('avg_memory_mb', 0),
            peak_memory_mb=monitor_summary.get('peak_memory_mb', 0),
            cpu_usage_percent=monitor_summary.get('avg_cpu_percent', 0),
            disk_io_read_mb=monitor_summary.get('final_disk_read_mb', 0),
            disk_io_write_mb=monitor_summary.get('final_disk_write_mb', 0),
            success_rate=success_rate,
            error_count=error_count,
            additional_metrics={
                'target_duration_minutes': duration_minutes,
                'actual_duration_minutes': actual_duration / 60,
                'target_ops_per_second': operations_per_second,
                'stability_score': max(0, 100 - (error_count / operations_count * 100)) if operations_count > 0 else 0
            }
        )
    
    def _generate_comprehensive_report(self, results: Dict[str, BenchmarkResult]):
        """生成综合性能报告"""
        report_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 计算总体统计
        total_operations = sum(r.operations_count for r in results.values())
        total_duration = sum(r.duration_seconds for r in results.values())
        avg_success_rate = sum(r.success_rate for r in results.values()) / len(results) if results else 0
        total_errors = sum(r.error_count for r in results.values())
        
        # 性能排名
        performance_ranking = sorted(
            results.items(),
            key=lambda x: x[1].operations_per_second,
            reverse=True
        )
        
        # 内存使用排名
        memory_ranking = sorted(
            results.items(),
            key=lambda x: x[1].peak_memory_mb,
            reverse=True
        )
        
        report_data = {
            'report_info': {
                'generated_at': datetime.now().isoformat(),
                'total_tests': len(results),
                'total_operations': total_operations,
                'total_duration_seconds': total_duration,
                'average_success_rate': avg_success_rate,
                'total_errors': total_errors
            },
            'test_results': {name: result.to_dict() for name, result in results.items()},
            'performance_ranking': [
                {
                    'test_name': name,
                    'operations_per_second': result.operations_per_second,
                    'rank': i + 1
                }
                for i, (name, result) in enumerate(performance_ranking)
            ],
            'memory_usage_ranking': [
                {
                    'test_name': name,
                    'peak_memory_mb': result.peak_memory_mb,
                    'rank': i + 1
                }
                for i, (name, result) in enumerate(memory_ranking)
            ],
            'recommendations': self._generate_recommendations(results)
        }
        
        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"[REPORT] 综合性能报告已生成: {report_file}")
        
        # 打印摘要
        self._print_report_summary(report_data)
    
    def _generate_recommendations(self, results: Dict[str, BenchmarkResult]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        # 分析性能瓶颈
        slowest_test = min(results.values(), key=lambda x: x.operations_per_second)
        if slowest_test.operations_per_second < 1000:
            recommendations.append(f"测试 '{slowest_test.test_name}' 性能较低 ({slowest_test.operations_per_second:.1f} ops/s)，建议优化I/O操作和缓冲策略")
        
        # 分析内存使用
        highest_memory = max(results.values(), key=lambda x: x.peak_memory_mb)
        if highest_memory.peak_memory_mb > 500:
            recommendations.append(f"测试 '{highest_memory.test_name}' 内存使用过高 ({highest_memory.peak_memory_mb:.1f} MB)，建议优化内存管理和对象复用")
        
        # 分析错误率
        high_error_tests = [r for r in results.values() if r.success_rate < 95]
        if high_error_tests:
            test_names = [t.test_name for t in high_error_tests]
            recommendations.append(f"测试 {test_names} 错误率较高，建议加强错误处理和重试机制")
        
        # 分析并发性能
        concurrent_test = results.get('concurrent_logging')
        if concurrent_test and concurrent_test.operations_per_second < 5000:
            recommendations.append("并发性能有待提升，建议优化锁机制和队列大小")
        
        # 分析磁盘I/O
        high_io_tests = [r for r in results.values() if r.disk_io_write_mb > 100]
        if high_io_tests:
            recommendations.append("磁盘I/O较高，建议启用压缩和优化文件轮转策略")
        
        if not recommendations:
            recommendations.append("所有测试性能表现良好，系统运行稳定")
        
        return recommendations
    
    def _print_report_summary(self, report_data: Dict[str, Any]):
        """打印报告摘要"""
        print("\n" + "="*80)
        print("性能基准测试报告摘要")
        print("="*80)
        
        info = report_data['report_info']
        print(f"测试时间: {info['generated_at']}")
        print(f"总测试数: {info['total_tests']}")
        print(f"总操作数: {info['total_operations']:,}")
        print(f"总耗时: {info['total_duration_seconds']:.2f} 秒")
        print(f"平均成功率: {info['average_success_rate']:.2f}%")
        print(f"总错误数: {info['total_errors']}")
        
        print("\n性能排名 (前5名):")
        for i, item in enumerate(report_data['performance_ranking'][:5]):
            print(f"  {i+1}. {item['test_name']}: {item['operations_per_second']:.1f} ops/s")
        
        print("\n内存使用排名 (前5名):")
        for i, item in enumerate(report_data['memory_usage_ranking'][:5]):
            print(f"  {i+1}. {item['test_name']}: {item['peak_memory_mb']:.1f} MB")
        
        print("\n优化建议:")
        for i, rec in enumerate(report_data['recommendations']):
            print(f"  {i+1}. {rec}")
        
        print("="*80)


def main():
    """主函数"""
    print("开始性能基准测试...")
    
    # 创建基准测试器
    benchmark = PerformanceBenchmark()
    
    try:
        # 运行所有测试
        results = benchmark.run_all_benchmarks()
        
        print(f"\n所有测试完成！共运行 {len(results)} 个测试")
        print(f"结果保存在: {benchmark.output_dir}")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()