"""
状态管理器（StateManager）

管理优化过程的状态保存和恢复功能，支持检查点管理和损坏状态检测。
包含高斯过程模型的序列化支持。
"""

import os
import json
import pickle
import hashlib
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import numpy as np

from autodl_core import OptimizationHistory, OptimizationResult, ParameterSpace
from gaussian_process import GaussianProcess


class CheckpointError(Exception):
    """检查点相关错误"""
    pass


class StateManager:
    """
    状态管理器
    
    负责优化过程中状态的保存、恢复和检查点管理
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints", 
                 max_checkpoints: int = 10,
                 compression: bool = True):
        """
        初始化状态管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大保留的检查点数量
            compression: 是否压缩保存的状态文件
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.compression = compression
        
        # 创建检查点目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger("StateManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"StateManager初始化完成，检查点目录: {self.checkpoint_dir}")
    
    def save_state(self, optimizer_state: Dict[str, Any], 
                   iteration: int, 
                   checkpoint_name: Optional[str] = None) -> str:
        """
        保存优化器状态到检查点文件
        
        Args:
            optimizer_state: 优化器状态字典，包含：
                - 'history': OptimizationHistory对象
                - 'parameter_space': ParameterSpace对象
                - 'gaussian_process': GaussianProcess对象
                - 'acquisition_function': 采集函数配置
                - 'config': 其他配置信息
            iteration: 当前迭代次数
            checkpoint_name: 检查点名称，如果为None则自动生成
            
        Returns:
            保存的检查点文件路径
            
        Raises:
            CheckpointError: 当保存失败时
        """
        try:
            # 生成检查点名称
            if checkpoint_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_iter_{iteration:04d}_{timestamp}"
            
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
            
            # 准备保存的状态数据
            state_data = {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'optimizer_state': {}
            }
            
            # 序列化各个组件
            for key, value in optimizer_state.items():
                if key == 'history' and isinstance(value, OptimizationHistory):
                    state_data['optimizer_state'][key] = value.to_dict()
                
                elif key == 'parameter_space' and isinstance(value, ParameterSpace):
                    state_data['optimizer_state'][key] = value.to_dict()
                
                elif key == 'gaussian_process' and isinstance(value, GaussianProcess):
                    # 高斯过程需要特殊处理
                    gp_data = self._serialize_gaussian_process(value)
                    state_data['optimizer_state'][key] = gp_data
                
                else:
                    # 其他数据直接保存
                    state_data['optimizer_state'][key] = value
            
            # 计算校验和
            state_data['checksum'] = self._calculate_checksum(state_data['optimizer_state'])
            
            # 保存到临时文件，然后原子性移动
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, 
                                           dir=self.checkpoint_dir, 
                                           suffix='.tmp') as tmp_file:
                if self.compression:
                    import gzip
                    with gzip.open(tmp_file, 'wb') as gz_file:
                        pickle.dump(state_data, gz_file, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(state_data, tmp_file, protocol=pickle.HIGHEST_PROTOCOL)
                
                tmp_path = tmp_file.name
            
            # 原子性移动到最终位置
            shutil.move(tmp_path, checkpoint_path)
            
            self.logger.info(f"状态保存成功: {checkpoint_path}")
            
            # 清理旧的检查点
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            # 清理临时文件
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            error_msg = f"保存状态失败: {str(e)}"
            self.logger.error(error_msg)
            raise CheckpointError(error_msg) from e
    
    def load_state(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        从检查点文件加载优化器状态
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            包含优化器状态的字典
            
        Raises:
            CheckpointError: 当加载失败或文件损坏时
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise CheckpointError(f"检查点文件不存在: {checkpoint_path}")
        
        try:
            # 验证检查点文件
            if not self.validate_checkpoint(str(checkpoint_path)):
                raise CheckpointError(f"检查点文件损坏或无效: {checkpoint_path}")
            
            # 加载状态数据
            if self.compression and checkpoint_path.suffix == '.pkl':
                # 尝试压缩格式
                try:
                    import gzip
                    with gzip.open(checkpoint_path, 'rb') as gz_file:
                        state_data = pickle.load(gz_file)
                except:
                    # 如果压缩格式失败，尝试普通格式
                    with open(checkpoint_path, 'rb') as f:
                        state_data = pickle.load(f)
            else:
                with open(checkpoint_path, 'rb') as f:
                    state_data = pickle.load(f)
            
            # 验证校验和
            stored_checksum = state_data.get('checksum')
            if stored_checksum:
                calculated_checksum = self._calculate_checksum(state_data['optimizer_state'])
                if stored_checksum != calculated_checksum:
                    raise CheckpointError("检查点文件校验和不匹配，文件可能已损坏")
            
            # 反序列化各个组件
            optimizer_state = {}
            for key, value in state_data['optimizer_state'].items():
                if key == 'history' and isinstance(value, dict):
                    optimizer_state[key] = OptimizationHistory.from_dict(value)
                
                elif key == 'parameter_space' and isinstance(value, dict):
                    optimizer_state[key] = ParameterSpace.from_dict(value)
                
                elif key == 'gaussian_process' and isinstance(value, dict):
                    optimizer_state[key] = self._deserialize_gaussian_process(value)
                
                else:
                    optimizer_state[key] = value
            
            # 添加元数据
            optimizer_state['_metadata'] = {
                'iteration': state_data.get('iteration'),
                'timestamp': state_data.get('timestamp'),
                'version': state_data.get('version'),
                'checkpoint_path': str(checkpoint_path)
            }
            
            self.logger.info(f"状态加载成功: {checkpoint_path}")
            return optimizer_state
            
        except Exception as e:
            error_msg = f"加载状态失败: {str(e)}"
            self.logger.error(error_msg)
            raise CheckpointError(error_msg) from e
    
    def create_checkpoint(self, optimizer_state: Dict[str, Any], 
                         iteration: int,
                         checkpoint_freq: int = 10) -> Optional[str]:
        """
        根据频率创建检查点
        
        Args:
            optimizer_state: 优化器状态
            iteration: 当前迭代次数
            checkpoint_freq: 检查点频率
            
        Returns:
            如果创建了检查点，返回文件路径；否则返回None
        """
        if iteration % checkpoint_freq == 0 or iteration == 1:
            return self.save_state(optimizer_state, iteration)
        return None
    
    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        验证检查点文件的完整性
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            True如果文件有效，False如果文件损坏
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            return False
        
        if checkpoint_path.stat().st_size == 0:
            return False
        
        try:
            # 尝试加载文件头部信息
            if self.compression:
                try:
                    import gzip
                    with gzip.open(checkpoint_path, 'rb') as gz_file:
                        # 只读取少量数据来验证文件格式
                        data = pickle.load(gz_file)
                except:
                    # 如果压缩格式失败，尝试普通格式
                    with open(checkpoint_path, 'rb') as f:
                        data = pickle.load(f)
            else:
                with open(checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
            
            # 检查必需的字段
            required_fields = ['optimizer_state', 'iteration', 'timestamp']
            for field in required_fields:
                if field not in data:
                    return False
            
            # 检查优化器状态的基本结构
            optimizer_state = data['optimizer_state']
            if not isinstance(optimizer_state, dict):
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"验证检查点文件失败: {checkpoint_path}, 错误: {e}")
            return False
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        列出所有可用的检查点
        
        Returns:
            检查点信息列表，每个元素包含文件路径、迭代次数、时间戳等信息
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                # 快速读取元数据
                if self.compression:
                    try:
                        import gzip
                        with gzip.open(checkpoint_file, 'rb') as gz_file:
                            data = pickle.load(gz_file)
                    except:
                        with open(checkpoint_file, 'rb') as f:
                            data = pickle.load(f)
                else:
                    with open(checkpoint_file, 'rb') as f:
                        data = pickle.load(f)
                
                checkpoint_info = {
                    'path': str(checkpoint_file),
                    'filename': checkpoint_file.name,
                    'iteration': data.get('iteration', 0),
                    'timestamp': data.get('timestamp', ''),
                    'version': data.get('version', 'unknown'),
                    'file_size': checkpoint_file.stat().st_size,
                    'is_valid': True
                }
                
                checkpoints.append(checkpoint_info)
                
            except Exception as e:
                # 如果无法读取，标记为无效
                checkpoint_info = {
                    'path': str(checkpoint_file),
                    'filename': checkpoint_file.name,
                    'iteration': -1,
                    'timestamp': '',
                    'version': 'unknown',
                    'file_size': checkpoint_file.stat().st_size,
                    'is_valid': False,
                    'error': str(e)
                }
                checkpoints.append(checkpoint_info)
        
        # 按迭代次数排序
        checkpoints.sort(key=lambda x: x['iteration'], reverse=True)
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        获取最新的有效检查点路径
        
        Returns:
            最新检查点的路径，如果没有有效检查点则返回None
        """
        checkpoints = self.list_checkpoints()
        
        for checkpoint in checkpoints:
            if checkpoint['is_valid']:
                return checkpoint['path']
        
        return None
    
    def cleanup_corrupted_checkpoints(self) -> int:
        """
        清理损坏的检查点文件
        
        Returns:
            清理的文件数量
        """
        checkpoints = self.list_checkpoints()
        cleaned_count = 0
        
        for checkpoint in checkpoints:
            if not checkpoint['is_valid']:
                try:
                    os.unlink(checkpoint['path'])
                    self.logger.info(f"清理损坏的检查点: {checkpoint['filename']}")
                    cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"清理检查点失败: {checkpoint['filename']}, 错误: {e}")
        
        return cleaned_count
    
    def _serialize_gaussian_process(self, gp: GaussianProcess) -> Dict[str, Any]:
        """
        序列化高斯过程模型
        
        Args:
            gp: GaussianProcess对象
            
        Returns:
            序列化后的字典
        """
        gp_data = {
            'model_info': gp.get_model_info(),
            'is_fitted': gp.is_fitted
        }
        
        if gp.is_fitted:
            # 保存训练数据
            X_train, y_train = gp.get_training_data()
            gp_data['X_train'] = X_train.tolist() if X_train is not None else None
            gp_data['y_train'] = y_train.tolist() if y_train is not None else None
            
            # 保存模型参数
            gp_data['hyperparameters'] = gp.get_hyperparameters()
            
            # 保存sklearn模型（使用pickle序列化）
            import io
            buffer = io.BytesIO()
            pickle.dump(gp.gp, buffer)
            gp_data['sklearn_model'] = buffer.getvalue()
        
        # 保存初始化参数
        gp_data['init_params'] = {
            'length_scale': gp.length_scale,
            'length_scale_bounds': gp.length_scale_bounds,
            'noise_level': gp.noise_level,
            'noise_level_bounds': gp.noise_level_bounds,
            'constant_value': gp.constant_value,
            'constant_value_bounds': gp.constant_value_bounds,
            'n_restarts_optimizer': gp.n_restarts_optimizer,
            'random_state': gp.random_state
        }
        
        return gp_data
    
    def _deserialize_gaussian_process(self, gp_data: Dict[str, Any]) -> GaussianProcess:
        """
        反序列化高斯过程模型
        
        Args:
            gp_data: 序列化的高斯过程数据
            
        Returns:
            重建的GaussianProcess对象
        """
        # 重建高斯过程对象
        init_params = gp_data['init_params']
        gp = GaussianProcess(**init_params)
        
        if gp_data['is_fitted']:
            # 恢复训练数据
            X_train = np.array(gp_data['X_train']) if gp_data['X_train'] else None
            y_train = np.array(gp_data['y_train']) if gp_data['y_train'] else None
            
            if X_train is not None and y_train is not None:
                # 恢复sklearn模型
                import io
                buffer = io.BytesIO(gp_data['sklearn_model'])
                gp.gp = pickle.load(buffer)
                
                # 恢复状态
                gp.X_train = X_train
                gp.y_train = y_train
                gp.is_fitted = True
                gp.n_observations = X_train.shape[0]
                
                # 恢复其他属性
                model_info = gp_data['model_info']
                if 'last_fit_time' in model_info and model_info['last_fit_time']:
                    gp.last_fit_time = datetime.fromisoformat(model_info['last_fit_time'])
                
                gp.log_marginal_likelihood = model_info.get('log_marginal_likelihood')
        
        return gp
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """
        计算数据的校验和
        
        Args:
            data: 要计算校验和的数据
            
        Returns:
            MD5校验和字符串
        """
        # 将数据转换为JSON字符串（排序键以确保一致性）
        json_str = json.dumps(data, sort_keys=True, default=str)
        
        # 计算MD5哈希
        return hashlib.md5(json_str.encode('utf-8')).hexdigest()
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点文件，保留最新的max_checkpoints个"""
        checkpoints = self.list_checkpoints()
        
        # 只保留有效的检查点
        valid_checkpoints = [cp for cp in checkpoints if cp['is_valid']]
        
        if len(valid_checkpoints) > self.max_checkpoints:
            # 删除最旧的检查点
            to_delete = valid_checkpoints[self.max_checkpoints:]
            
            for checkpoint in to_delete:
                try:
                    os.unlink(checkpoint['path'])
                    self.logger.info(f"清理旧检查点: {checkpoint['filename']}")
                except Exception as e:
                    self.logger.warning(f"清理检查点失败: {checkpoint['filename']}, 错误: {e}")
    
    def export_checkpoint(self, checkpoint_path: str, export_path: str) -> None:
        """
        导出检查点到指定路径
        
        Args:
            checkpoint_path: 源检查点路径
            export_path: 导出目标路径
            
        Raises:
            CheckpointError: 当导出失败时
        """
        try:
            if not self.validate_checkpoint(checkpoint_path):
                raise CheckpointError(f"源检查点文件无效: {checkpoint_path}")
            
            shutil.copy2(checkpoint_path, export_path)
            self.logger.info(f"检查点导出成功: {checkpoint_path} -> {export_path}")
            
        except Exception as e:
            error_msg = f"导出检查点失败: {str(e)}"
            self.logger.error(error_msg)
            raise CheckpointError(error_msg) from e
    
    def import_checkpoint(self, import_path: str, checkpoint_name: Optional[str] = None) -> str:
        """
        导入外部检查点文件
        
        Args:
            import_path: 要导入的检查点文件路径
            checkpoint_name: 导入后的检查点名称，如果为None则使用原文件名
            
        Returns:
            导入后的检查点路径
            
        Raises:
            CheckpointError: 当导入失败时
        """
        try:
            if not os.path.exists(import_path):
                raise CheckpointError(f"导入文件不存在: {import_path}")
            
            # 验证导入文件
            if not self.validate_checkpoint(import_path):
                raise CheckpointError(f"导入文件无效或损坏: {import_path}")
            
            # 确定目标文件名
            if checkpoint_name is None:
                checkpoint_name = Path(import_path).stem
            
            target_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
            
            # 复制文件
            shutil.copy2(import_path, target_path)
            
            self.logger.info(f"检查点导入成功: {import_path} -> {target_path}")
            return str(target_path)
            
        except Exception as e:
            error_msg = f"导入检查点失败: {str(e)}"
            self.logger.error(error_msg)
            raise CheckpointError(error_msg) from e
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        获取检查点的详细信息
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            检查点详细信息字典
            
        Raises:
            CheckpointError: 当读取失败时
        """
        try:
            if not self.validate_checkpoint(checkpoint_path):
                raise CheckpointError(f"检查点文件无效: {checkpoint_path}")
            
            # 加载检查点数据
            if self.compression:
                try:
                    import gzip
                    with gzip.open(checkpoint_path, 'rb') as gz_file:
                        data = pickle.load(gz_file)
                except:
                    with open(checkpoint_path, 'rb') as f:
                        data = pickle.load(f)
            else:
                with open(checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
            
            # 提取详细信息
            info = {
                'file_path': checkpoint_path,
                'file_size': os.path.getsize(checkpoint_path),
                'iteration': data.get('iteration'),
                'timestamp': data.get('timestamp'),
                'version': data.get('version'),
                'checksum': data.get('checksum'),
                'components': list(data['optimizer_state'].keys())
            }
            
            # 添加组件特定信息
            optimizer_state = data['optimizer_state']
            
            if 'history' in optimizer_state:
                history_data = optimizer_state['history']
                info['history_info'] = {
                    'total_iterations': history_data.get('total_iterations', 0),
                    'total_time': history_data.get('total_time', 0.0),
                    'task_type': history_data.get('task_type', 'unknown'),
                    'acquisition_function': history_data.get('acquisition_function', 'unknown'),
                    'best_objective_value': None
                }
                
                if history_data.get('best_result'):
                    info['history_info']['best_objective_value'] = history_data['best_result'].get('objective_value')
            
            if 'parameter_space' in optimizer_state:
                param_space_data = optimizer_state['parameter_space']
                if 'parameters' in param_space_data:
                    info['parameter_space_info'] = {
                        'parameter_count': len(param_space_data['parameters']),
                        'parameter_names': list(param_space_data['parameters'].keys())
                    }
            
            if 'gaussian_process' in optimizer_state:
                gp_data = optimizer_state['gaussian_process']
                info['gaussian_process_info'] = {
                    'is_fitted': gp_data.get('is_fitted', False),
                    'n_observations': 0
                }
                
                if gp_data.get('X_train'):
                    info['gaussian_process_info']['n_observations'] = len(gp_data['X_train'])
            
            return info
            
        except Exception as e:
            error_msg = f"获取检查点信息失败: {str(e)}"
            self.logger.error(error_msg)
            raise CheckpointError(error_msg) from e


def create_default_state_manager(checkpoint_dir: str = "checkpoints") -> StateManager:
    """
    创建默认配置的状态管理器
    
    Args:
        checkpoint_dir: 检查点目录
        
    Returns:
        配置好的StateManager实例
    """
    return StateManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=10,
        compression=True
    )


if __name__ == "__main__":
    # 测试代码
    print("测试状态管理器...")
    
    # 创建测试数据
    from autodl_core import create_default_parameter_space, OptimizationHistory, OptimizationResult
    from gaussian_process import create_default_gaussian_process
    import numpy as np
    
    # 创建状态管理器
    state_manager = create_default_state_manager("test_checkpoints")
    
    # 创建测试的优化器状态
    parameter_space = create_default_parameter_space()
    history = OptimizationHistory()
    gp = create_default_gaussian_process(random_state=42)
    
    # 添加一些测试数据到高斯过程
    X_test = np.random.uniform(-1, 1, (5, 10))
    y_test = np.random.uniform(0, 1, 5)
    gp.fit(X_test, y_test)
    
    # 添加一些测试结果到历史
    for i in range(3):
        params = parameter_space.sample_random_parameters(seed=42+i)
        result = OptimizationResult(
            parameters=params,
            objective_value=0.8 + 0.1 * i,
            metrics={'AUROC': 0.8 + 0.1 * i, 'AUPRC': 0.75 + 0.1 * i},
            iteration=i+1,
            timestamp=datetime.now(),
            evaluation_time=120.0
        )
        history.add_result(result)
    
    optimizer_state = {
        'history': history,
        'parameter_space': parameter_space,
        'gaussian_process': gp,
        'acquisition_function': {'type': 'EI', 'xi': 0.01},
        'config': {'max_iterations': 100, 'task_type': 'LDA'}
    }
    
    # 测试保存状态
    print("测试保存状态...")
    checkpoint_path = state_manager.save_state(optimizer_state, iteration=3)
    print(f"状态保存到: {checkpoint_path}")
    
    # 测试验证检查点
    print("测试验证检查点...")
    is_valid = state_manager.validate_checkpoint(checkpoint_path)
    print(f"检查点验证结果: {is_valid}")
    
    # 测试加载状态
    print("测试加载状态...")
    loaded_state = state_manager.load_state(checkpoint_path)
    print(f"加载的状态包含组件: {list(loaded_state.keys())}")
    
    # 验证加载的数据
    loaded_history = loaded_state['history']
    loaded_gp = loaded_state['gaussian_process']
    
    print(f"加载的历史记录: {loaded_history.total_iterations} 次迭代")
    print(f"加载的高斯过程: {loaded_gp.n_observations} 个观测点")
    print(f"最佳目标值: {loaded_history.get_best_objective_value()}")
    
    # 测试列出检查点
    print("测试列出检查点...")
    checkpoints = state_manager.list_checkpoints()
    for cp in checkpoints:
        print(f"  {cp['filename']}: 迭代{cp['iteration']}, 大小{cp['file_size']}字节")
    
    # 测试获取检查点信息
    print("测试获取检查点详细信息...")
    info = state_manager.get_checkpoint_info(checkpoint_path)
    print(f"检查点信息: {info}")
    
    # 清理测试文件
    import shutil
    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints")
    
    print("状态管理器测试完成!")