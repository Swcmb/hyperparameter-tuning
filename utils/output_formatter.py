"""
输出格式化器 (OutputFormatter)

负责格式化不同类型的输出内容，包括训练信息、优化信息、系统信息等。
与task_evaluator.py保持一致的格式化风格，移除所有emoji表情符号。
"""

import re
import time
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class ProgressInfo:
    """进度信息数据结构"""
    current: int
    total: int
    start_time: datetime
    message: str = ""
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
    
    @property
    def percentage(self) -> float:
        """计算完成百分比"""
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100.0)
    
    @property
    def elapsed_time(self) -> timedelta:
        """计算已用时间"""
        return datetime.now() - self.start_time
    
    @property
    def estimated_remaining(self) -> Optional[timedelta]:
        """估算剩余时间"""
        if self.current <= 0 or self.current >= self.total:
            return None
        
        elapsed = self.elapsed_time.total_seconds()
        rate = self.current / elapsed
        remaining_items = self.total - self.current
        remaining_seconds = remaining_items / rate
        
        return timedelta(seconds=remaining_seconds)


class OutputFormatter:
    """
    输出格式化器
    
    负责格式化不同类型的输出内容，确保与task_evaluator.py的风格一致
    """
    
    # Emoji检测正则表达式
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    
    # Emoji到文本的映射
    EMOJI_REPLACEMENTS = {
        "🚀": "[启动]",
        "⚡": "[快速]",
        "🔥": "[热门]",
        "✅": "[完成]",
        "❌": "[错误]",
        "⚠️": "[警告]",
        "📊": "[统计]",
        "📈": "[上升]",
        "📉": "[下降]",
        "🎯": "[目标]",
        "🔍": "[搜索]",
        "💡": "[提示]",
        "🛠️": "[工具]",
        "⏰": "[时间]",
        "📝": "[记录]",
        "🔧": "[配置]",
        "📋": "[列表]",
        "🎉": "[成功]",
        "💾": "[保存]",
        "🔄": "[更新]",
        "⭐": "[重要]",
        "🎲": "[随机]",
        "🧠": "[智能]",
        "🔬": "[实验]",
        "📦": "[包]",
        "🌟": "[优秀]",
        "⚙️": "[设置]",
        "📁": "[文件夹]",
        "📄": "[文件]",
        "🔗": "[链接]",
        "🎨": "[样式]",
        "🏆": "[最佳]",
        "🎪": "[展示]",
        "🔮": "[预测]",
        "🎭": "[模式]",
        "🎪": "[演示]"
    }
    
    def __init__(self):
        """初始化输出格式化器"""
        pass
    
    def remove_emojis(self, text: str) -> str:
        """
        移除文本中的emoji表情符号
        
        Args:
            text: 输入文本
            
        Returns:
            移除emoji后的文本
        """
        if not isinstance(text, str):
            return str(text)
        
        # 首先尝试替换已知的emoji
        result = text
        for emoji, replacement in self.EMOJI_REPLACEMENTS.items():
            if emoji in result:
                result = result.replace(emoji, replacement)
        
        # 然后移除剩余的emoji（使用更全面的正则表达式）
        # 扩展的emoji正则表达式
        emoji_pattern = re.compile(
            "["
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        
        result = emoji_pattern.sub('', result)
        
        # 清理多余的空格
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def format_training_info(self, 
                           epoch: int, 
                           total_epochs: int,
                           batch: Optional[int] = None, 
                           total_batches: Optional[int] = None,
                           loss_info: Optional[Dict[str, float]] = None, 
                           metrics: Optional[Dict[str, float]] = None,
                           timing_info: Optional[Dict[str, float]] = None,
                           memory_info: Optional[Dict[str, float]] = None) -> List[str]:
        """
        格式化训练信息，与task_evaluator.py保持一致的风格
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
            batch: 当前batch（可选）
            total_batches: 总batch数（可选）
            loss_info: 损失信息字典
            metrics: 性能指标字典
            timing_info: 时间信息字典
            memory_info: 内存信息字典
            
        Returns:
            格式化后的信息行列表
        """
        lines = []
        
        # Epoch信息
        if batch is not None and total_batches is not None:
            # 批次级别的信息
            progress_percent = (batch / total_batches) * 100 if total_batches > 0 else 0
            lines.append(f"[BATCH] Epoch {epoch:02d} | Batch {batch:04d}/{total_batches:04d} ({progress_percent:5.1f}%)")
        else:
            # Epoch级别的信息
            lines.append(f"[EPOCH {epoch:02d}/{total_epochs}] ========== 训练轮次完成 ==========")
        
        # 损失信息
        if loss_info:
            if batch is not None:
                # 批次损失信息
                loss_parts = []
                if 'total' in loss_info:
                    loss_parts.append(f"总计={loss_info['total']:.6f}")
                if 'bce' in loss_info:
                    loss_parts.append(f"BCE={loss_info['bce']:.6f}")
                if 'contrast' in loss_info:
                    loss_parts.append(f"对比={loss_info['contrast']:.6f}")
                if 'adversarial' in loss_info:
                    loss_parts.append(f"对抗={loss_info['adversarial']:.6f}")
                
                if loss_parts:
                    lines.append(f"[BATCH] 当前批次损失: {', '.join(loss_parts)}")
                
                if 'avg_total' in loss_info:
                    lines.append(f"[BATCH] 累计平均损失: {loss_info['avg_total']:.6f}")
            else:
                # Epoch损失统计
                lines.append(f"[EPOCH {epoch:02d}/{total_epochs}] 损失统计:")
                if 'total' in loss_info and 'total_std' in loss_info:
                    lines.append(f"[EPOCH {epoch:02d}/{total_epochs}]   总损失: {loss_info['total']:.6f} ± {loss_info['total_std']:.6f}")
                if 'bce' in loss_info and 'bce_std' in loss_info:
                    lines.append(f"[EPOCH {epoch:02d}/{total_epochs}]   BCE损失: {loss_info['bce']:.6f} ± {loss_info['bce_std']:.6f}")
                if 'contrast' in loss_info and 'contrast_std' in loss_info:
                    lines.append(f"[EPOCH {epoch:02d}/{total_epochs}]   对比学习损失: {loss_info['contrast']:.6f} ± {loss_info['contrast_std']:.6f}")
                if 'adversarial' in loss_info and 'adversarial_std' in loss_info:
                    lines.append(f"[EPOCH {epoch:02d}/{total_epochs}]   节点对抗损失: {loss_info['adversarial']:.6f} ± {loss_info['adversarial_std']:.6f}")
        
        # 性能指标
        if metrics:
            if batch is None:  # 只在epoch级别显示详细指标
                lines.append(f"[EPOCH {epoch:02d}/{total_epochs}] 性能指标:")
                if 'auroc' in metrics:
                    lines.append(f"[EPOCH {epoch:02d}/{total_epochs}]   AUROC: {metrics['auroc']:.6f}")
                if 'auprc' in metrics:
                    lines.append(f"[EPOCH {epoch:02d}/{total_epochs}]   AUPRC: {metrics['auprc']:.6f}")
                if 'f1' in metrics:
                    lines.append(f"[EPOCH {epoch:02d}/{total_epochs}]   F1-Score: {metrics['f1']:.6f}")
        
        # 时间信息
        if timing_info:
            if batch is not None and 'batch_time' in timing_info:
                lines.append(f"[BATCH] 批次处理时间: {timing_info['batch_time']:.4f}秒")
            elif 'epoch_time' in timing_info:
                lines.append(f"[EPOCH {epoch:02d}/{total_epochs}] 训练时间: {timing_info['epoch_time']:.2f}秒")
                if 'avg_batch_time' in timing_info:
                    lines.append(f"[EPOCH {epoch:02d}/{total_epochs}] 平均批次处理时间: {timing_info['avg_batch_time']:.4f}秒")
                if 'estimated_remaining' in timing_info:
                    lines.append(f"[PROGRESS] 预计剩余时间: {timing_info['estimated_remaining']:.1f}秒 ({timing_info['estimated_remaining']/60:.1f}分钟)")
        
        # 内存信息
        if memory_info:
            if 'gpu_allocated' in memory_info:
                if batch is not None:
                    lines.append(f"[BATCH] GPU内存使用: {memory_info['gpu_allocated']:.3f} GB")
                else:
                    lines.append(f"[MEMORY] GPU内存清理后使用量: {memory_info['gpu_allocated']:.3f} GB")
        
        return lines
    
    def format_optimization_info(self, 
                               iteration: int,
                               suggested_params: Dict[str, Any],
                               acquisition_value: Optional[float] = None,
                               gp_stats: Optional[Dict[str, Any]] = None,
                               evaluation_result: Optional[Dict[str, float]] = None,
                               timing_info: Optional[Dict[str, float]] = None) -> List[str]:
        """
        格式化优化信息
        
        Args:
            iteration: 优化迭代次数
            suggested_params: 建议的参数
            acquisition_value: 采集函数值
            gp_stats: 高斯过程统计信息
            evaluation_result: 评估结果
            timing_info: 时间信息
            
        Returns:
            格式化后的信息行列表
        """
        lines = []
        
        # 优化迭代信息
        lines.append(f"[OPTIMIZATION] ========== 迭代 {iteration} ==========")
        
        # 参数建议
        lines.append("[SUGGESTION] 建议参数:")
        for param_name, param_value in suggested_params.items():
            if isinstance(param_value, float):
                lines.append(f"[SUGGESTION]   {param_name}: {param_value:.6f}")
            else:
                lines.append(f"[SUGGESTION]   {param_name}: {param_value}")
        
        # 采集函数信息
        if acquisition_value is not None:
            lines.append(f"[ACQUISITION] 采集函数值: {acquisition_value:.6f}")
        
        # 高斯过程统计
        if gp_stats:
            lines.append("[GP_STATS] 高斯过程统计:")
            if 'mean_prediction' in gp_stats:
                lines.append(f"[GP_STATS]   预测均值: {gp_stats['mean_prediction']:.6f}")
            if 'std_prediction' in gp_stats:
                lines.append(f"[GP_STATS]   预测标准差: {gp_stats['std_prediction']:.6f}")
            if 'confidence_interval' in gp_stats:
                ci = gp_stats['confidence_interval']
                lines.append(f"[GP_STATS]   95%置信区间: [{ci[0]:.6f}, {ci[1]:.6f}]")
            if 'hyperparameters' in gp_stats:
                lines.append("[GP_STATS]   超参数:")
                for hp_name, hp_value in gp_stats['hyperparameters'].items():
                    lines.append(f"[GP_STATS]     {hp_name}: {hp_value:.6f}")
        
        # 评估结果
        if evaluation_result:
            lines.append("[EVALUATION] 评估结果:")
            if 'AUROC' in evaluation_result:
                lines.append(f"[EVALUATION]   AUROC: {evaluation_result['AUROC']:.6f}")
            if 'AUPRC' in evaluation_result:
                lines.append(f"[EVALUATION]   AUPRC: {evaluation_result['AUPRC']:.6f}")
            if 'F1' in evaluation_result:
                lines.append(f"[EVALUATION]   F1-Score: {evaluation_result['F1']:.6f}")
            if 'loss' in evaluation_result:
                lines.append(f"[EVALUATION]   损失: {evaluation_result['loss']:.6f}")
        
        # 时间信息
        if timing_info:
            if 'evaluation_time' in timing_info:
                lines.append(f"[TIMING] 评估时间: {timing_info['evaluation_time']:.2f}秒")
            if 'total_time' in timing_info:
                lines.append(f"[TIMING] 总用时: {timing_info['total_time']:.2f}秒")
        
        return lines
    
    def format_system_info(self, 
                          gpu_info: Optional[Dict[str, Any]] = None,
                          memory_info: Optional[Dict[str, Any]] = None,
                          cpu_info: Optional[Dict[str, Any]] = None,
                          disk_info: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        格式化系统信息
        
        Args:
            gpu_info: GPU信息字典
            memory_info: 内存信息字典
            cpu_info: CPU信息字典
            disk_info: 磁盘信息字典
            
        Returns:
            格式化后的信息行列表
        """
        lines = []
        
        # GPU信息
        if gpu_info:
            lines.append("[GPU] GPU设备信息:")
            if 'name' in gpu_info:
                lines.append(f"[GPU] GPU设备: {gpu_info['name']}")
            if 'total_memory' in gpu_info:
                lines.append(f"[GPU] GPU总内存: {gpu_info['total_memory']:.2f} GB")
            if 'allocated_memory' in gpu_info:
                lines.append(f"[GPU] 已分配内存: {gpu_info['allocated_memory']:.2f} GB")
            if 'cached_memory' in gpu_info:
                lines.append(f"[GPU] 缓存内存: {gpu_info['cached_memory']:.2f} GB")
            if 'utilization' in gpu_info:
                lines.append(f"[GPU] GPU利用率: {gpu_info['utilization']:.1f}%")
        
        # 内存信息
        if memory_info:
            lines.append("[MEMORY] 系统内存信息:")
            if 'total' in memory_info:
                lines.append(f"[MEMORY] 总内存: {memory_info['total']:.2f} GB")
            if 'available' in memory_info:
                lines.append(f"[MEMORY] 可用内存: {memory_info['available']:.2f} GB")
            if 'used' in memory_info:
                lines.append(f"[MEMORY] 已用内存: {memory_info['used']:.2f} GB")
            if 'percent' in memory_info:
                lines.append(f"[MEMORY] 内存使用率: {memory_info['percent']:.1f}%")
        
        # CPU信息
        if cpu_info:
            lines.append("[CPU] CPU信息:")
            if 'usage' in cpu_info:
                lines.append(f"[CPU] CPU使用率: {cpu_info['usage']:.1f}%")
            if 'cores' in cpu_info:
                lines.append(f"[CPU] CPU核心数: {cpu_info['cores']}")
            if 'frequency' in cpu_info:
                lines.append(f"[CPU] CPU频率: {cpu_info['frequency']:.2f} GHz")
        
        # 磁盘信息
        if disk_info:
            lines.append("[DISK] 磁盘信息:")
            if 'total' in disk_info:
                lines.append(f"[DISK] 总空间: {disk_info['total']:.2f} GB")
            if 'used' in disk_info:
                lines.append(f"[DISK] 已用空间: {disk_info['used']:.2f} GB")
            if 'free' in disk_info:
                lines.append(f"[DISK] 可用空间: {disk_info['free']:.2f} GB")
            if 'percent' in disk_info:
                lines.append(f"[DISK] 磁盘使用率: {disk_info['percent']:.1f}%")
        
        return lines
    
    def format_error_info(self, 
                         error: Exception, 
                         context: Optional[Dict[str, Any]] = None,
                         component: Optional[str] = None,
                         operation: Optional[str] = None) -> List[str]:
        """
        格式化错误信息
        
        Args:
            error: 异常对象
            context: 错误上下文信息
            component: 出错的组件名称
            operation: 出错的操作名称
            
        Returns:
            格式化后的错误信息行列表
        """
        lines = []
        
        # 错误标题
        error_type = type(error).__name__
        if component and operation:
            lines.append(f"[ERROR] {component} - {operation} 失败: {error_type}")
        elif component:
            lines.append(f"[ERROR] {component} 错误: {error_type}")
        else:
            lines.append(f"[ERROR] 系统错误: {error_type}")
        
        # 错误消息
        error_msg = str(error)
        if error_msg:
            # 移除可能的emoji
            error_msg = self.remove_emojis(error_msg)
            lines.append(f"[ERROR] 错误信息: {error_msg}")
        
        # 上下文信息
        if context:
            lines.append("[ERROR] 错误上下文:")
            for key, value in context.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"[ERROR]   {key}: {str(value)[:100]}...")
                else:
                    lines.append(f"[ERROR]   {key}: {value}")
        
        # 错误时间
        lines.append(f"[ERROR] 发生时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return lines
    
    def format_progress_bar(self, 
                           current: int, 
                           total: int, 
                           width: int = 50,
                           prefix: str = "",
                           suffix: str = "",
                           fill_char: str = "█",
                           empty_char: str = "░") -> str:
        """
        格式化进度条
        
        Args:
            current: 当前进度
            total: 总数
            width: 进度条宽度
            prefix: 前缀文本
            suffix: 后缀文本
            fill_char: 填充字符
            empty_char: 空白字符
            
        Returns:
            格式化后的进度条字符串
        """
        if total <= 0:
            return f"{prefix} [{'?' * width}] 0/0 (0.0%) {suffix}"
        
        percentage = min(100.0, (current / total) * 100.0)
        filled_length = int(width * current // total)
        
        bar = fill_char * filled_length + empty_char * (width - filled_length)
        
        return f"{prefix} [{bar}] {current}/{total} ({percentage:.1f}%) {suffix}"
    
    def format_progress_info(self, progress: ProgressInfo) -> List[str]:
        """
        格式化进度信息
        
        Args:
            progress: 进度信息对象
            
        Returns:
            格式化后的进度信息行列表
        """
        lines = []
        
        # 进度条
        progress_bar = self.format_progress_bar(
            progress.current, 
            progress.total,
            prefix="[PROGRESS]"
        )
        lines.append(progress_bar)
        
        # 时间信息
        elapsed = progress.elapsed_time
        elapsed_str = f"{elapsed.total_seconds():.1f}秒"
        if elapsed.total_seconds() > 60:
            elapsed_str += f" ({elapsed.total_seconds()/60:.1f}分钟)"
        
        lines.append(f"[PROGRESS] 已用时间: {elapsed_str}")
        
        # 剩余时间估算
        if progress.estimated_remaining:
            remaining = progress.estimated_remaining
            remaining_str = f"{remaining.total_seconds():.1f}秒"
            if remaining.total_seconds() > 60:
                remaining_str += f" ({remaining.total_seconds()/60:.1f}分钟)"
            lines.append(f"[PROGRESS] 预计剩余: {remaining_str}")
        
        # 处理速度
        if progress.current > 0:
            rate = progress.current / progress.elapsed_time.total_seconds()
            lines.append(f"[PROGRESS] 处理速度: {rate:.2f} 项/秒")
        
        # 附加消息
        if progress.message:
            lines.append(f"[PROGRESS] {progress.message}")
        
        # 附加指标
        if progress.metrics:
            for metric_name, metric_value in progress.metrics.items():
                if isinstance(metric_value, float):
                    lines.append(f"[PROGRESS] {metric_name}: {metric_value:.4f}")
                else:
                    lines.append(f"[PROGRESS] {metric_name}: {metric_value}")
        
        return lines
    
    def format_results_summary(self, 
                             results: Dict[str, Any],
                             title: str = "结果摘要") -> List[str]:
        """
        格式化结果摘要
        
        Args:
            results: 结果字典
            title: 摘要标题
            
        Returns:
            格式化后的结果摘要行列表
        """
        lines = []
        
        # 标题
        separator = "=" * 80
        lines.append(separator)
        lines.append(f"[RESULTS] ========== {title} ==========")
        
        # 主要性能指标
        if any(key in results for key in ['AUROC', 'AUPRC', 'F1', 'auroc', 'auprc', 'f1']):
            lines.append("[RESULTS] === 主要性能指标 ===")
            
            auroc = results.get('AUROC') or results.get('auroc')
            if auroc is not None:
                lines.append(f"[RESULTS] AUROC (Area Under ROC Curve): {auroc:.6f}")
            
            auprc = results.get('AUPRC') or results.get('auprc')
            if auprc is not None:
                lines.append(f"[RESULTS] AUPRC (Area Under Precision-Recall Curve): {auprc:.6f}")
            
            f1 = results.get('F1') or results.get('f1')
            if f1 is not None:
                lines.append(f"[RESULTS] F1-Score: {f1:.6f}")
            
            lines.append("")
        
        # 分类性能指标
        classification_metrics = ['accuracy', 'precision', 'recall', 'specificity']
        if any(key in results for key in classification_metrics):
            lines.append("[RESULTS] === 分类性能指标 ===")
            
            for metric in classification_metrics:
                if metric in results:
                    value = results[metric]
                    metric_name = {
                        'accuracy': '准确率 (Accuracy)',
                        'precision': '精确率 (Precision)',
                        'recall': '召回率 (Recall/Sensitivity)',
                        'specificity': '特异性 (Specificity)'
                    }.get(metric, metric)
                    
                    lines.append(f"[RESULTS] {metric_name}: {value:.6f} ({value*100:.2f}%)")
            
            lines.append("")
        
        # 混淆矩阵
        if 'confusion_matrix' in results or 'cm' in results:
            cm = results.get('confusion_matrix') or results.get('cm')
            if cm and len(cm) == 4:
                tn, fp, fn, tp = cm
                lines.append("[RESULTS] === 混淆矩阵分析 ===")
                lines.append(f"[RESULTS] 真负例 (True Negatives): {tn}")
                lines.append(f"[RESULTS] 假正例 (False Positives): {fp}")
                lines.append(f"[RESULTS] 假负例 (False Negatives): {fn}")
                lines.append(f"[RESULTS] 真正例 (True Positives): {tp}")
                lines.append(f"[RESULTS] 总样本数: {tn + fp + fn + tp}")
                lines.append("")
        
        # 训练信息
        if 'loss' in results or 'training_time' in results:
            lines.append("[RESULTS] === 训练信息 ===")
            
            if 'loss' in results:
                lines.append(f"[RESULTS] 最终训练损失: {results['loss']:.6f}")
            
            if 'training_time' in results:
                training_time = results['training_time']
                lines.append(f"[RESULTS] 训练时间: {training_time:.2f}秒 ({training_time/60:.2f}分钟)")
            
            lines.append("")
        
        # 性能评估
        if auroc is not None:
            lines.append("[RESULTS] === 模型性能评估 ===")
            if auroc >= 0.9:
                performance_level = "优秀"
            elif auroc >= 0.8:
                performance_level = "良好"
            elif auroc >= 0.7:
                performance_level = "中等"
            else:
                performance_level = "需要改进"
            
            lines.append(f"[RESULTS] 模型性能等级: {performance_level} (基于AUROC)")
        
        lines.append(separator)
        
        return lines
    
    def format_configuration_info(self, 
                                config: Dict[str, Any],
                                title: str = "配置信息") -> List[str]:
        """
        格式化配置信息
        
        Args:
            config: 配置字典
            title: 配置标题
            
        Returns:
            格式化后的配置信息行列表
        """
        lines = []
        
        lines.append(f"[CONFIG] ========== {title} ==========")
        
        # 按类别组织配置项
        categories = {
            'model': ['model_type', 'hidden1', 'hidden2', 'dimensions', 'decoder1'],
            'training': ['lr', 'batch', 'epochs', 'dropout', 'weight_decay'],
            'loss': ['loss_ratio1', 'loss_ratio2', 'loss_ratio3', 'alpha', 'beta', 'gamma'],
            'attention': ['gat_heads', 'gt_heads', 'fusion_heads', 'fusion_strategy'],
            'moco': ['moco_type', 'moco_K', 'moco_m', 'moco_T', 'moco_tau1', 'moco_tau2'],
            'system': ['device', 'cuda', 'seed', 'threads']
        }
        
        for category, keys in categories.items():
            category_items = {k: v for k, v in config.items() if k in keys}
            if category_items:
                category_name = {
                    'model': '模型结构',
                    'training': '训练参数',
                    'loss': '损失函数',
                    'attention': '注意力机制',
                    'moco': 'MoCo参数',
                    'system': '系统设置'
                }.get(category, category)
                
                lines.append(f"[CONFIG] === {category_name} ===")
                for key, value in category_items.items():
                    if isinstance(value, float):
                        lines.append(f"[CONFIG]   {key}: {value:.8f}")
                    else:
                        lines.append(f"[CONFIG]   {key}: {value}")
                lines.append("")
        
        # 其他未分类的配置项
        categorized_keys = set()
        for keys in categories.values():
            categorized_keys.update(keys)
        
        other_items = {k: v for k, v in config.items() if k not in categorized_keys}
        if other_items:
            lines.append("[CONFIG] === 其他配置 ===")
            for key, value in other_items.items():
                if isinstance(value, float):
                    lines.append(f"[CONFIG]   {key}: {value:.8f}")
                else:
                    lines.append(f"[CONFIG]   {key}: {value}")
        
        return lines
    
    def format_data_statistics(self, 
                             data_stats: Dict[str, Any]) -> List[str]:
        """
        格式化数据统计信息
        
        Args:
            data_stats: 数据统计字典
            
        Returns:
            格式化后的数据统计信息行列表
        """
        lines = []
        
        lines.append("[DATA_STATS] ========== 数据统计 ==========")
        
        # 样本统计
        if 'total_samples' in data_stats:
            lines.append(f"[DATA_STATS] 总样本数量: {data_stats['total_samples']}")
        
        if 'positive_samples' in data_stats and 'negative_samples' in data_stats:
            pos = data_stats['positive_samples']
            neg = data_stats['negative_samples']
            total = pos + neg
            pos_ratio = (pos / total * 100) if total > 0 else 0
            
            lines.append(f"[DATA_STATS] 正样本数量: {pos}")
            lines.append(f"[DATA_STATS] 负样本数量: {neg}")
            lines.append(f"[DATA_STATS] 正样本比例: {pos_ratio:.2f}%")
            lines.append(f"[DATA_STATS] 数据平衡性: {'平衡' if 40 <= pos_ratio <= 60 else '不平衡'}")
        
        # 特征统计
        if 'feature_dim' in data_stats:
            lines.append(f"[DATA_STATS] 特征维度: {data_stats['feature_dim']}")
        
        if 'node_count' in data_stats:
            lines.append(f"[DATA_STATS] 节点数量: {data_stats['node_count']}")
        
        if 'edge_count' in data_stats:
            lines.append(f"[DATA_STATS] 边数量: {data_stats['edge_count']}")
        
        # 数据质量
        if 'missing_values' in data_stats:
            lines.append(f"[DATA_STATS] 缺失值数量: {data_stats['missing_values']}")
        
        if 'data_quality' in data_stats:
            lines.append(f"[DATA_STATS] 数据质量: {data_stats['data_quality']}")
        
        return lines


def test_output_formatter():
    """测试输出格式化器的功能"""
    print("测试OutputFormatter...")
    
    formatter = OutputFormatter()
    
    # 测试emoji移除
    test_text = "🚀 开始训练 ✅ 完成 ❌ 错误 📊 统计"
    cleaned_text = formatter.remove_emojis(test_text)
    print(f"Emoji移除测试:")
    print(f"  原文: {test_text}")
    print(f"  结果: {cleaned_text}")
    
    # 测试单个emoji替换
    print("\n单个emoji测试:")
    for emoji, replacement in list(formatter.EMOJI_REPLACEMENTS.items())[:5]:
        test_single = f"测试 {emoji} 符号"
        result_single = formatter.remove_emojis(test_single)
        print(f"  {test_single} -> {result_single}")
    
    # 测试训练信息格式化
    print("\n训练信息格式化测试:")
    training_lines = formatter.format_training_info(
        epoch=1,
        total_epochs=50,
        batch=10,
        total_batches=100,
        loss_info={
            'total': 0.5234,
            'bce': 0.3456,
            'contrast': 0.1234,
            'adversarial': 0.0544,
            'avg_total': 0.5123
        },
        timing_info={'batch_time': 0.1234},
        memory_info={'gpu_allocated': 2.345}
    )
    for line in training_lines:
        print(line)
    
    # 测试优化信息格式化
    print("\n优化信息格式化测试:")
    opt_lines = formatter.format_optimization_info(
        iteration=5,
        suggested_params={
            'lr': 0.001234,
            'hidden1': 128,
            'batch': 32
        },
        acquisition_value=0.8765,
        evaluation_result={
            'AUROC': 0.8234,
            'AUPRC': 0.7654,
            'F1': 0.7123
        }
    )
    for line in opt_lines:
        print(line)
    
    # 测试进度条
    print("\n进度条测试:")
    progress_bar = formatter.format_progress_bar(75, 100, prefix="训练进度")
    print(progress_bar)
    
    print("\nOutputFormatter功能测试完成")


if __name__ == "__main__":
    test_output_formatter()