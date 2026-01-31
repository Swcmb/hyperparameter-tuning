"""
文件管理器 (FileManager)

负责管理日志文件的创建、轮转、压缩和清理功能。
提供磁盘空间监控和自动清理机制。
"""

import os
import shutil
import gzip
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging


@dataclass
class FileInfo:
    """文件信息数据结构"""
    path: Path
    size: int
    created_time: datetime
    modified_time: datetime
    file_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'path': str(self.path),
            'size': self.size,
            'created_time': self.created_time.isoformat(),
            'modified_time': self.modified_time.isoformat(),
            'file_type': self.file_type
        }


@dataclass
class DiskSpaceInfo:
    """磁盘空间信息"""
    total: int
    used: int
    free: int
    
    @property
    def usage_percent(self) -> float:
        """使用率百分比"""
        return (self.used / self.total * 100) if self.total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'total_gb': self.total / (1024**3),
            'used_gb': self.used / (1024**3),
            'free_gb': self.free / (1024**3),
            'usage_percent': self.usage_percent
        }


class FileManager:
    """
    文件管理器
    
    负责日志文件的创建、轮转、压缩和清理
    """
    
    def __init__(self, 
                 base_dir: str = "logs",
                 run_name: Optional[str] = None,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 max_files_per_type: int = 10,
                 cleanup_days: int = 30,
                 disk_usage_threshold: float = 85.0,
                 enable_compression: bool = True):
        """
        初始化文件管理器
        
        Args:
            base_dir: 基础目录路径
            run_name: 运行名称，如果为None则自动生成
            max_file_size: 单个文件最大大小（字节）
            max_files_per_type: 每种类型文件的最大数量
            cleanup_days: 清理多少天前的文件
            disk_usage_threshold: 磁盘使用率阈值（百分比）
            enable_compression: 是否启用压缩
        """
        self.base_dir = Path(base_dir)
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.max_file_size = max_file_size
        self.max_files_per_type = max_files_per_type
        self.cleanup_days = cleanup_days
        self.disk_usage_threshold = disk_usage_threshold
        self.enable_compression = enable_compression
        
        # 创建运行目录
        self.run_dir = self.base_dir / f"{self.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件类型定义
        self.file_types = {
            'main': 'main.log',
            'training': 'training.log',
            'optimization': 'optimization.log',
            'system': 'system.log',
            'error': 'error.log',
            'debug': 'debug.log',
            'structured': 'structured.jsonl',
            'metrics': 'metrics.jsonl',
            'config': 'config.json'
        }
        
        # 活动文件句柄
        self._file_handles: Dict[str, Any] = {}
        self._file_locks: Dict[str, threading.Lock] = {}
        
        # 初始化锁
        for file_type in self.file_types.keys():
            self._file_locks[file_type] = threading.Lock()
        
        # 设置日志
        self.logger = logging.getLogger(f"FileManager_{self.run_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"[INIT] FileManager初始化完成")
        self.logger.info(f"[INIT] 运行目录: {self.run_dir}")
        self.logger.info(f"[INIT] 最大文件大小: {self.max_file_size / (1024*1024):.1f} MB")
        self.logger.info(f"[INIT] 清理阈值: {self.cleanup_days} 天")
        self.logger.info(f"[INIT] 磁盘使用率阈值: {self.disk_usage_threshold}%")
    
    def create_log_files(self) -> Dict[str, Path]:
        """
        创建各种类型的日志文件
        
        Returns:
            文件类型到路径的映射字典
        """
        created_files = {}
        
        for file_type, filename in self.file_types.items():
            file_path = self.run_dir / filename
            
            try:
                # 创建文件（如果不存在）
                if not file_path.exists():
                    file_path.touch()
                    self.logger.info(f"[CREATE] 创建日志文件: {file_path}")
                
                created_files[file_type] = file_path
                
                # 写入文件头信息
                self._write_file_header(file_path, file_type)
                
            except Exception as e:
                self.logger.error(f"[ERROR] 创建文件失败 {file_path}: {e}")
                continue
        
        self.logger.info(f"[CREATE] 成功创建 {len(created_files)} 个日志文件")
        return created_files
    
    def _write_file_header(self, file_path: Path, file_type: str):
        """
        写入文件头信息
        
        Args:
            file_path: 文件路径
            file_type: 文件类型
        """
        try:
            header_info = {
                'file_type': file_type,
                'created_time': datetime.now().isoformat(),
                'run_name': self.run_name,
                'version': '1.0'
            }
            
            if file_type == 'config':
                # JSON配置文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(header_info, f, indent=2, ensure_ascii=False)
            elif file_type in ['structured', 'metrics']:
                # JSONL文件
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(header_info, ensure_ascii=False) + '\n')
            else:
                # 普通日志文件
                header_text = (
                    f"# {file_type.upper()} LOG FILE\n"
                    f"# Created: {header_info['created_time']}\n"
                    f"# Run: {header_info['run_name']}\n"
                    f"# Version: {header_info['version']}\n"
                    f"{'='*80}\n\n"
                )
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(header_text)
                    
        except Exception as e:
            self.logger.error(f"[ERROR] 写入文件头失败 {file_path}: {e}")
    
    def get_file_handle(self, file_type: str, mode: str = 'a') -> Optional[Any]:
        """
        获取文件句柄（线程安全）
        
        Args:
            file_type: 文件类型
            mode: 打开模式
            
        Returns:
            文件句柄或None
        """
        if file_type not in self.file_types:
            self.logger.error(f"[ERROR] 未知文件类型: {file_type}")
            return None
        
        with self._file_locks[file_type]:
            # 检查是否需要轮转
            if file_type in self._file_handles:
                current_file = self.run_dir / self.file_types[file_type]
                if self._should_rotate_file(current_file):
                    self._rotate_file(file_type)
            
            # 获取或创建文件句柄
            if file_type not in self._file_handles:
                file_path = self.run_dir / self.file_types[file_type]
                try:
                    self._file_handles[file_type] = open(file_path, mode, encoding='utf-8')
                except Exception as e:
                    self.logger.error(f"[ERROR] 打开文件失败 {file_path}: {e}")
                    return None
            
            return self._file_handles[file_type]
    
    def write_to_file(self, 
                     file_type: str, 
                     content: str, 
                     flush: bool = True) -> bool:
        """
        写入内容到指定类型的文件
        
        Args:
            file_type: 文件类型
            content: 要写入的内容
            flush: 是否立即刷新到磁盘
            
        Returns:
            是否写入成功
        """
        file_handle = self.get_file_handle(file_type)
        if not file_handle:
            return False
        
        try:
            with self._file_locks[file_type]:
                # 添加时间戳
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                timestamped_content = f"[{timestamp}] {content}\n"
                
                file_handle.write(timestamped_content)
                
                if flush:
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                
                return True
                
        except Exception as e:
            self.logger.error(f"[ERROR] 写入文件失败 {file_type}: {e}")
            return False
    
    def write_structured_data(self, 
                            file_type: str, 
                            data: Dict[str, Any],
                            flush: bool = True) -> bool:
        """
        写入结构化数据到JSONL文件
        
        Args:
            file_type: 文件类型（应该是structured或metrics）
            data: 要写入的数据字典
            flush: 是否立即刷新到磁盘
            
        Returns:
            是否写入成功
        """
        if file_type not in ['structured', 'metrics']:
            self.logger.error(f"[ERROR] 文件类型 {file_type} 不支持结构化数据")
            return False
        
        file_handle = self.get_file_handle(file_type)
        if not file_handle:
            return False
        
        try:
            with self._file_locks[file_type]:
                # 添加时间戳
                data_with_timestamp = {
                    'timestamp': datetime.now().isoformat(),
                    **data
                }
                
                json_line = json.dumps(data_with_timestamp, ensure_ascii=False)
                file_handle.write(json_line + '\n')
                
                if flush:
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                
                return True
                
        except Exception as e:
            self.logger.error(f"[ERROR] 写入结构化数据失败 {file_type}: {e}")
            return False
    
    def _should_rotate_file(self, file_path: Path) -> bool:
        """
        检查文件是否需要轮转
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否需要轮转
        """
        try:
            if not file_path.exists():
                return False
            
            file_size = file_path.stat().st_size
            return file_size >= self.max_file_size
            
        except Exception as e:
            self.logger.error(f"[ERROR] 检查文件大小失败 {file_path}: {e}")
            return False
    
    def _rotate_file(self, file_type: str) -> bool:
        """
        轮转指定类型的文件
        
        Args:
            file_type: 文件类型
            
        Returns:
            是否轮转成功
        """
        try:
            # 关闭当前文件句柄
            if file_type in self._file_handles:
                self._file_handles[file_type].close()
                del self._file_handles[file_type]
            
            current_file = self.run_dir / self.file_types[file_type]
            if not current_file.exists():
                return False
            
            # 生成轮转后的文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = current_file.stem
            extension = current_file.suffix
            rotated_name = f"{base_name}_{timestamp}{extension}"
            rotated_file = self.run_dir / rotated_name
            
            # 移动文件
            shutil.move(str(current_file), str(rotated_file))
            self.logger.info(f"[ROTATE] 文件轮转: {current_file.name} -> {rotated_name}")
            
            # 压缩文件（如果启用）
            if self.enable_compression:
                self._compress_file(rotated_file)
            
            # 清理旧文件
            self._cleanup_old_files(file_type)
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] 文件轮转失败 {file_type}: {e}")
            return False
    
    def rotate_log_file(self, file_path: Path, max_size: Optional[int] = None) -> bool:
        """
        手动轮转指定的日志文件
        
        Args:
            file_path: 文件路径
            max_size: 最大文件大小，如果为None则使用默认值
            
        Returns:
            是否轮转成功
        """
        max_size = max_size or self.max_file_size
        
        try:
            if not file_path.exists():
                self.logger.warning(f"[WARNING] 文件不存在: {file_path}")
                return False
            
            file_size = file_path.stat().st_size
            if file_size < max_size:
                self.logger.info(f"[INFO] 文件大小未达到轮转阈值: {file_size} < {max_size}")
                return False
            
            # 查找对应的文件类型
            file_type = None
            for ft, filename in self.file_types.items():
                if file_path.name == filename:
                    file_type = ft
                    break
            
            if file_type:
                return self._rotate_file(file_type)
            else:
                # 直接轮转文件
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_name = file_path.stem
                extension = file_path.suffix
                rotated_name = f"{base_name}_{timestamp}{extension}"
                rotated_file = file_path.parent / rotated_name
                
                shutil.move(str(file_path), str(rotated_file))
                self.logger.info(f"[ROTATE] 手动轮转: {file_path.name} -> {rotated_name}")
                
                if self.enable_compression:
                    self._compress_file(rotated_file)
                
                return True
                
        except Exception as e:
            self.logger.error(f"[ERROR] 手动轮转失败 {file_path}: {e}")
            return False
    
    def _compress_file(self, file_path: Path) -> bool:
        """
        压缩文件
        
        Args:
            file_path: 要压缩的文件路径
            
        Returns:
            是否压缩成功
        """
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # 删除原文件
            file_path.unlink()
            
            original_size = file_path.stat().st_size if file_path.exists() else 0
            compressed_size = compressed_path.stat().st_size
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            self.logger.info(f"[COMPRESS] 文件压缩完成: {file_path.name} -> {compressed_path.name}")
            self.logger.info(f"[COMPRESS] 压缩率: {compression_ratio:.1f}% ({original_size} -> {compressed_size} bytes)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] 文件压缩失败 {file_path}: {e}")
            return False
    
    def _cleanup_old_files(self, file_type: str):
        """
        清理指定类型的旧文件
        
        Args:
            file_type: 文件类型
        """
        try:
            base_name = Path(self.file_types[file_type]).stem
            pattern = f"{base_name}_*"
            
            # 查找所有相关文件
            matching_files = []
            for file_path in self.run_dir.glob(pattern):
                if file_path.is_file():
                    file_info = FileInfo(
                        path=file_path,
                        size=file_path.stat().st_size,
                        created_time=datetime.fromtimestamp(file_path.stat().st_ctime),
                        modified_time=datetime.fromtimestamp(file_path.stat().st_mtime),
                        file_type=file_type
                    )
                    matching_files.append(file_info)
            
            # 按修改时间排序（最新的在前）
            matching_files.sort(key=lambda x: x.modified_time, reverse=True)
            
            # 保留最新的N个文件，删除其余的
            files_to_delete = matching_files[self.max_files_per_type:]
            
            for file_info in files_to_delete:
                try:
                    file_info.path.unlink()
                    self.logger.info(f"[CLEANUP] 删除旧文件: {file_info.path.name}")
                except Exception as e:
                    self.logger.error(f"[ERROR] 删除文件失败 {file_info.path}: {e}")
            
            if files_to_delete:
                self.logger.info(f"[CLEANUP] 清理完成，删除了 {len(files_to_delete)} 个旧文件")
                
        except Exception as e:
            self.logger.error(f"[ERROR] 清理旧文件失败 {file_type}: {e}")
    
    def cleanup_old_logs(self, keep_days: Optional[int] = None) -> int:
        """
        清理旧的日志文件
        
        Args:
            keep_days: 保留多少天的文件，如果为None则使用默认值
            
        Returns:
            删除的文件数量
        """
        keep_days = keep_days or self.cleanup_days
        cutoff_time = datetime.now() - timedelta(days=keep_days)
        deleted_count = 0
        
        self.logger.info(f"[CLEANUP] 开始清理 {keep_days} 天前的日志文件")
        self.logger.info(f"[CLEANUP] 截止时间: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 遍历基础目录下的所有运行目录
            for run_dir in self.base_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                # 检查目录的修改时间
                dir_mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
                if dir_mtime < cutoff_time:
                    try:
                        # 计算目录中的文件数量
                        file_count = len(list(run_dir.glob('*')))
                        
                        # 删除整个目录
                        shutil.rmtree(run_dir)
                        deleted_count += file_count
                        
                        self.logger.info(f"[CLEANUP] 删除旧运行目录: {run_dir.name} ({file_count} 个文件)")
                        
                    except Exception as e:
                        self.logger.error(f"[ERROR] 删除目录失败 {run_dir}: {e}")
                        continue
            
            self.logger.info(f"[CLEANUP] 清理完成，总共删除了 {deleted_count} 个文件")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"[ERROR] 清理日志失败: {e}")
            return deleted_count
    
    def get_disk_space_info(self, path: Optional[Path] = None) -> DiskSpaceInfo:
        """
        获取磁盘空间信息
        
        Args:
            path: 检查的路径，如果为None则检查运行目录
            
        Returns:
            磁盘空间信息
        """
        check_path = path or self.run_dir
        
        try:
            if os.name == 'nt':  # Windows
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                total_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(str(check_path)),
                    ctypes.pointer(free_bytes),
                    ctypes.pointer(total_bytes),
                    None
                )
                free = free_bytes.value
                total = total_bytes.value
                used = total - free
            else:  # Unix/Linux
                statvfs = os.statvfs(check_path)
                total = statvfs.f_frsize * statvfs.f_blocks
                free = statvfs.f_frsize * statvfs.f_available
                used = total - free
            
            return DiskSpaceInfo(total=total, used=used, free=free)
            
        except Exception as e:
            self.logger.error(f"[ERROR] 获取磁盘空间信息失败: {e}")
            # 返回默认值
            return DiskSpaceInfo(total=0, used=0, free=0)
    
    def monitor_disk_space(self) -> bool:
        """
        监控磁盘空间使用情况
        
        Returns:
            磁盘空间是否充足
        """
        disk_info = self.get_disk_space_info()
        
        if disk_info.usage_percent > self.disk_usage_threshold:
            self.logger.warning(f"[WARNING] 磁盘空间不足!")
            self.logger.warning(f"[WARNING] 使用率: {disk_info.usage_percent:.1f}% (阈值: {self.disk_usage_threshold}%)")
            self.logger.warning(f"[WARNING] 可用空间: {disk_info.free / (1024**3):.2f} GB")
            
            # 尝试自动清理
            self.logger.info("[AUTO_CLEANUP] 尝试自动清理旧文件...")
            deleted_count = self.cleanup_old_logs()
            
            if deleted_count > 0:
                # 重新检查磁盘空间
                new_disk_info = self.get_disk_space_info()
                self.logger.info(f"[AUTO_CLEANUP] 清理后磁盘使用率: {new_disk_info.usage_percent:.1f}%")
                return new_disk_info.usage_percent <= self.disk_usage_threshold
            else:
                return False
        
        return True
    
    def get_file_statistics(self) -> Dict[str, Any]:
        """
        获取文件统计信息
        
        Returns:
            文件统计信息字典
        """
        stats = {
            'run_dir': str(self.run_dir),
            'total_files': 0,
            'total_size': 0,
            'file_types': {},
            'disk_info': self.get_disk_space_info().to_dict(),
            'created_time': datetime.now().isoformat()
        }
        
        try:
            for file_path in self.run_dir.rglob('*'):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_ext = file_path.suffix.lower()
                    
                    stats['total_files'] += 1
                    stats['total_size'] += file_size
                    
                    if file_ext not in stats['file_types']:
                        stats['file_types'][file_ext] = {
                            'count': 0,
                            'total_size': 0
                        }
                    
                    stats['file_types'][file_ext]['count'] += 1
                    stats['file_types'][file_ext]['total_size'] += file_size
            
            # 转换大小为可读格式
            stats['total_size_mb'] = stats['total_size'] / (1024 * 1024)
            
            for ext_info in stats['file_types'].values():
                ext_info['size_mb'] = ext_info['total_size'] / (1024 * 1024)
            
        except Exception as e:
            self.logger.error(f"[ERROR] 获取文件统计失败: {e}")
        
        return stats
    
    def export_file_list(self, output_file: Optional[Path] = None) -> bool:
        """
        导出文件列表到JSON文件
        
        Args:
            output_file: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            是否导出成功
        """
        output_file = output_file or (self.run_dir / 'file_list.json')
        
        try:
            file_list = []
            
            for file_path in self.run_dir.rglob('*'):
                if file_path.is_file():
                    file_info = FileInfo(
                        path=file_path.relative_to(self.run_dir),
                        size=file_path.stat().st_size,
                        created_time=datetime.fromtimestamp(file_path.stat().st_ctime),
                        modified_time=datetime.fromtimestamp(file_path.stat().st_mtime),
                        file_type=file_path.suffix.lower()
                    )
                    file_list.append(file_info.to_dict())
            
            # 按修改时间排序
            file_list.sort(key=lambda x: x['modified_time'], reverse=True)
            
            export_data = {
                'export_time': datetime.now().isoformat(),
                'run_name': self.run_name,
                'run_dir': str(self.run_dir),
                'total_files': len(file_list),
                'files': file_list
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"[EXPORT] 文件列表导出完成: {output_file}")
            self.logger.info(f"[EXPORT] 总文件数: {len(file_list)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] 导出文件列表失败: {e}")
            return False
    
    def close_all_handles(self):
        """关闭所有文件句柄"""
        for file_type, file_handle in self._file_handles.items():
            try:
                if file_handle and not file_handle.closed:
                    file_handle.close()
                    self.logger.info(f"[CLOSE] 关闭文件句柄: {file_type}")
            except Exception as e:
                self.logger.error(f"[ERROR] 关闭文件句柄失败 {file_type}: {e}")
        
        self._file_handles.clear()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close_all_handles()
    
    def __del__(self):
        """析构函数"""
        try:
            self.close_all_handles()
        except:
            pass


def test_file_manager():
    """测试文件管理器的功能"""
    print("测试FileManager...")
    
    # 创建测试目录
    test_dir = Path("test_logs")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    try:
        with FileManager(base_dir=str(test_dir), run_name="test_run") as fm:
            print(f"运行目录: {fm.run_dir}")
            
            # 测试创建日志文件
            print("\n测试创建日志文件:")
            created_files = fm.create_log_files()
            for file_type, file_path in created_files.items():
                print(f"  {file_type}: {file_path}")
            
            # 测试写入文件
            print("\n测试写入文件:")
            success = fm.write_to_file('main', '这是一条测试日志消息')
            print(f"写入主日志: {'成功' if success else '失败'}")
            
            success = fm.write_to_file('training', '训练开始 - Epoch 1/10')
            print(f"写入训练日志: {'成功' if success else '失败'}")
            
            # 测试写入结构化数据
            print("\n测试写入结构化数据:")
            test_data = {
                'epoch': 1,
                'loss': 0.5234,
                'accuracy': 0.8765,
                'metrics': {'auroc': 0.9123, 'f1': 0.8456}
            }
            success = fm.write_structured_data('metrics', test_data)
            print(f"写入指标数据: {'成功' if success else '失败'}")
            
            # 测试磁盘空间监控
            print("\n测试磁盘空间监控:")
            disk_info = fm.get_disk_space_info()
            print(f"磁盘总空间: {disk_info.total / (1024**3):.2f} GB")
            print(f"已用空间: {disk_info.used / (1024**3):.2f} GB")
            print(f"可用空间: {disk_info.free / (1024**3):.2f} GB")
            print(f"使用率: {disk_info.usage_percent:.1f}%")
            
            # 测试文件统计
            print("\n测试文件统计:")
            stats = fm.get_file_statistics()
            print(f"总文件数: {stats['total_files']}")
            print(f"总大小: {stats['total_size_mb']:.2f} MB")
            print("文件类型分布:")
            for ext, info in stats['file_types'].items():
                print(f"  {ext or '无扩展名'}: {info['count']} 个文件, {info['size_mb']:.2f} MB")
            
            # 测试导出文件列表
            print("\n测试导出文件列表:")
            success = fm.export_file_list()
            print(f"导出文件列表: {'成功' if success else '失败'}")
            
        print("\nFileManager功能测试完成")
        
        # 清理测试目录
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("清理测试目录完成")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_file_manager()