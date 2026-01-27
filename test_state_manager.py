"""
状态管理器（StateManager）测试

测试状态保存、恢复、检查点管理和损坏状态检测功能
"""

import pytest
import tempfile
import shutil
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np

from state_manager import StateManager, CheckpointError, create_default_state_manager
from autodl_core import (
    OptimizationHistory, OptimizationResult, ParameterSpace, 
    create_default_parameter_space
)
from gaussian_process import GaussianProcess, create_default_gaussian_process


class TestStateManager:
    """测试StateManager基本功能"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.state_manager = StateManager(
            checkpoint_dir=self.temp_dir,
            max_checkpoints=5,
            compression=True
        )
        
        # 创建测试数据
        self.parameter_space = create_default_parameter_space()
        self.history = OptimizationHistory()
        self.gp = create_default_gaussian_process(random_state=42)
        
        # 添加测试数据到高斯过程
        X_test = np.random.uniform(-1, 1, (3, 10))
        y_test = np.random.uniform(0, 1, 3)
        self.gp.fit(X_test, y_test)
        
        # 添加测试结果到历史
        for i in range(2):
            params = self.parameter_space.sample_random_parameters(seed=42+i)
            result = OptimizationResult(
                parameters=params,
                objective_value=0.8 + 0.1 * i,
                metrics={'AUROC': 0.8 + 0.1 * i, 'AUPRC': 0.75 + 0.1 * i},
                iteration=i+1,
                timestamp=datetime.now(),
                evaluation_time=120.0
            )
            self.history.add_result(result)
        
        self.optimizer_state = {
            'history': self.history,
            'parameter_space': self.parameter_space,
            'gaussian_process': self.gp,
            'acquisition_function': {'type': 'EI', 'xi': 0.01},
            'config': {'max_iterations': 100, 'task_type': 'LDA'}
        }
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_state_manager_creation(self):
        """测试状态管理器创建"""
        assert self.state_manager.checkpoint_dir.exists()
        assert self.state_manager.max_checkpoints == 5
        assert self.state_manager.compression == True
    
    def test_save_and_load_state(self):
        """测试状态保存和加载"""
        # 保存状态
        checkpoint_path = self.state_manager.save_state(self.optimizer_state, iteration=5)
        
        assert os.path.exists(checkpoint_path)
        assert "checkpoint_iter_0005" in checkpoint_path
        
        # 加载状态
        loaded_state = self.state_manager.load_state(checkpoint_path)
        
        # 验证加载的数据
        assert 'history' in loaded_state
        assert 'parameter_space' in loaded_state
        assert 'gaussian_process' in loaded_state
        assert 'acquisition_function' in loaded_state
        assert 'config' in loaded_state
        assert '_metadata' in loaded_state
        
        # 验证历史记录
        loaded_history = loaded_state['history']
        assert isinstance(loaded_history, OptimizationHistory)
        assert loaded_history.total_iterations == 2
        assert loaded_history.get_best_objective_value() == 0.9
        
        # 验证参数空间
        loaded_space = loaded_state['parameter_space']
        assert isinstance(loaded_space, ParameterSpace)
        assert len(loaded_space.parameters) == len(self.parameter_space.parameters)
        
        # 验证高斯过程
        loaded_gp = loaded_state['gaussian_process']
        assert isinstance(loaded_gp, GaussianProcess)
        assert loaded_gp.is_fitted == True
        assert loaded_gp.n_observations == 3
        
        # 验证元数据
        metadata = loaded_state['_metadata']
        assert metadata['iteration'] == 5
        assert 'timestamp' in metadata
        assert metadata['version'] == '1.0'
    
    def test_checkpoint_validation(self):
        """测试检查点验证"""
        # 保存有效检查点
        checkpoint_path = self.state_manager.save_state(self.optimizer_state, iteration=1)
        
        # 验证有效检查点
        assert self.state_manager.validate_checkpoint(checkpoint_path) == True
        
        # 测试不存在的文件
        assert self.state_manager.validate_checkpoint("nonexistent.pkl") == False
        
        # 创建损坏的检查点文件
        corrupted_path = os.path.join(self.temp_dir, "corrupted.pkl")
        with open(corrupted_path, 'w') as f:
            f.write("invalid data")
        
        assert self.state_manager.validate_checkpoint(corrupted_path) == False
        
        # 创建空文件
        empty_path = os.path.join(self.temp_dir, "empty.pkl")
        Path(empty_path).touch()
        
        assert self.state_manager.validate_checkpoint(empty_path) == False
    
    def test_create_checkpoint_with_frequency(self):
        """测试按频率创建检查点"""
        # 测试应该创建检查点的情况
        checkpoint_path = self.state_manager.create_checkpoint(
            self.optimizer_state, iteration=10, checkpoint_freq=5
        )
        assert checkpoint_path is not None
        assert os.path.exists(checkpoint_path)
        
        # 测试不应该创建检查点的情况
        checkpoint_path = self.state_manager.create_checkpoint(
            self.optimizer_state, iteration=7, checkpoint_freq=5
        )
        assert checkpoint_path is None
        
        # 测试第一次迭代总是创建检查点
        checkpoint_path = self.state_manager.create_checkpoint(
            self.optimizer_state, iteration=1, checkpoint_freq=10
        )
        assert checkpoint_path is not None
    
    def test_list_checkpoints(self):
        """测试列出检查点"""
        # 创建多个检查点
        paths = []
        for i in range(3):
            path = self.state_manager.save_state(self.optimizer_state, iteration=i+1)
            paths.append(path)
        
        # 列出检查点
        checkpoints = self.state_manager.list_checkpoints()
        
        assert len(checkpoints) == 3
        
        # 验证检查点信息
        for cp in checkpoints:
            assert 'path' in cp
            assert 'filename' in cp
            assert 'iteration' in cp
            assert 'timestamp' in cp
            assert 'is_valid' in cp
            assert cp['is_valid'] == True
        
        # 验证按迭代次数降序排列
        iterations = [cp['iteration'] for cp in checkpoints]
        assert iterations == sorted(iterations, reverse=True)
    
    def test_get_latest_checkpoint(self):
        """测试获取最新检查点"""
        # 没有检查点时应该返回None
        assert self.state_manager.get_latest_checkpoint() is None
        
        # 创建多个检查点
        for i in range(3):
            self.state_manager.save_state(self.optimizer_state, iteration=i+1)
        
        # 获取最新检查点
        latest_path = self.state_manager.get_latest_checkpoint()
        assert latest_path is not None
        assert "checkpoint_iter_0003" in latest_path
    
    def test_cleanup_corrupted_checkpoints(self):
        """测试清理损坏的检查点"""
        # 创建有效检查点
        valid_path = self.state_manager.save_state(self.optimizer_state, iteration=1)
        
        # 创建损坏的检查点文件
        corrupted_path = os.path.join(self.temp_dir, "checkpoint_corrupted.pkl")
        with open(corrupted_path, 'w') as f:
            f.write("invalid data")
        
        # 清理前应该有2个文件
        all_files = list(Path(self.temp_dir).glob("*.pkl"))
        assert len(all_files) == 2
        
        # 清理损坏的检查点
        cleaned_count = self.state_manager.cleanup_corrupted_checkpoints()
        assert cleaned_count == 1
        
        # 清理后应该只有1个有效文件
        all_files = list(Path(self.temp_dir).glob("*.pkl"))
        assert len(all_files) == 1
        assert os.path.exists(valid_path)
    
    def test_checkpoint_info(self):
        """测试获取检查点详细信息"""
        checkpoint_path = self.state_manager.save_state(self.optimizer_state, iteration=5)
        
        info = self.state_manager.get_checkpoint_info(checkpoint_path)
        
        # 验证基本信息
        assert info['iteration'] == 5
        assert info['version'] == '1.0'
        assert 'checksum' in info
        assert 'components' in info
        
        # 验证组件信息
        expected_components = ['history', 'parameter_space', 'gaussian_process', 
                              'acquisition_function', 'config']
        for component in expected_components:
            assert component in info['components']
        
        # 验证历史信息
        assert 'history_info' in info
        history_info = info['history_info']
        assert history_info['total_iterations'] == 2
        assert history_info['task_type'] == 'LDA'
        assert history_info['best_objective_value'] == 0.9
        
        # 验证参数空间信息
        assert 'parameter_space_info' in info
        param_info = info['parameter_space_info']
        assert param_info['parameter_count'] == 18
        assert 'dimensions' in param_info['parameter_names']
        
        # 验证高斯过程信息
        assert 'gaussian_process_info' in info
        gp_info = info['gaussian_process_info']
        assert gp_info['is_fitted'] == True
        assert gp_info['n_observations'] == 3
    
    def test_export_import_checkpoint(self):
        """测试检查点导出和导入"""
        # 创建检查点
        checkpoint_path = self.state_manager.save_state(self.optimizer_state, iteration=3)
        
        # 导出检查点
        export_path = os.path.join(self.temp_dir, "exported_checkpoint.pkl")
        self.state_manager.export_checkpoint(checkpoint_path, export_path)
        
        assert os.path.exists(export_path)
        assert self.state_manager.validate_checkpoint(export_path)
        
        # 创建新的状态管理器（模拟不同环境）
        new_temp_dir = tempfile.mkdtemp()
        try:
            new_state_manager = StateManager(checkpoint_dir=new_temp_dir)
            
            # 导入检查点
            imported_path = new_state_manager.import_checkpoint(
                export_path, "imported_checkpoint"
            )
            
            assert os.path.exists(imported_path)
            assert new_state_manager.validate_checkpoint(imported_path)
            
            # 验证导入的数据
            loaded_state = new_state_manager.load_state(imported_path)
            assert loaded_state['_metadata']['iteration'] == 3
            
        finally:
            shutil.rmtree(new_temp_dir)
    
    def test_max_checkpoints_cleanup(self):
        """测试最大检查点数量限制"""
        # 创建超过最大数量的检查点
        for i in range(8):  # 超过max_checkpoints=5
            self.state_manager.save_state(self.optimizer_state, iteration=i+1)
        
        # 验证只保留了最新的5个检查点
        checkpoints = self.state_manager.list_checkpoints()
        valid_checkpoints = [cp for cp in checkpoints if cp['is_valid']]
        
        assert len(valid_checkpoints) == 5
        
        # 验证保留的是最新的检查点
        iterations = [cp['iteration'] for cp in valid_checkpoints]
        assert min(iterations) == 4  # 应该保留迭代4-8
        assert max(iterations) == 8
    
    def test_gaussian_process_serialization(self):
        """测试高斯过程序列化和反序列化"""
        # 创建更复杂的高斯过程
        gp = create_default_gaussian_process(random_state=123)
        X = np.random.uniform(-2, 2, (10, 5))
        y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(10)
        gp.fit(X, y)
        
        # 获取预测结果（用于后续比较）
        X_test = np.random.uniform(-2, 2, (3, 5))
        mean_before, std_before = gp.predict(X_test)
        hyperparams_before = gp.get_hyperparameters()
        
        # 保存包含高斯过程的状态
        state_with_gp = {
            'gaussian_process': gp,
            'test_data': {'X_test': X_test}
        }
        
        checkpoint_path = self.state_manager.save_state(state_with_gp, iteration=1)
        
        # 加载状态
        loaded_state = self.state_manager.load_state(checkpoint_path)
        loaded_gp = loaded_state['gaussian_process']
        
        # 验证高斯过程状态
        assert loaded_gp.is_fitted == True
        assert loaded_gp.n_observations == 10
        
        # 验证预测结果一致性
        mean_after, std_after = loaded_gp.predict(X_test)
        np.testing.assert_array_almost_equal(mean_before, mean_after, decimal=6)
        np.testing.assert_array_almost_equal(std_before, std_after, decimal=6)
        
        # 验证超参数一致性
        hyperparams_after = loaded_gp.get_hyperparameters()
        for key in hyperparams_before:
            if key != 'log_marginal_likelihood':  # 这个可能有微小差异
                assert abs(hyperparams_before[key] - hyperparams_after[key]) < 1e-10
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试加载不存在的文件
        with pytest.raises(CheckpointError):
            self.state_manager.load_state("nonexistent.pkl")
        
        # 测试加载损坏的文件
        corrupted_path = os.path.join(self.temp_dir, "corrupted.pkl")
        with open(corrupted_path, 'w') as f:
            f.write("invalid data")
        
        with pytest.raises(CheckpointError):
            self.state_manager.load_state(corrupted_path)
        
        # 测试导出不存在的检查点
        with pytest.raises(CheckpointError):
            self.state_manager.export_checkpoint("nonexistent.pkl", "export.pkl")
        
        # 测试导入不存在的文件
        with pytest.raises(CheckpointError):
            self.state_manager.import_checkpoint("nonexistent.pkl")
        
        # 测试获取损坏文件的信息
        with pytest.raises(CheckpointError):
            self.state_manager.get_checkpoint_info(corrupted_path)


class TestStateManagerIntegration:
    """测试StateManager集成功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.state_manager = create_default_state_manager(self.temp_dir)
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_optimization_workflow_simulation(self):
        """模拟完整的优化工作流程"""
        # 初始化组件
        parameter_space = create_default_parameter_space()
        history = OptimizationHistory()
        gp = create_default_gaussian_process(random_state=42)
        
        # 模拟优化过程
        for iteration in range(1, 6):
            # 生成参数
            params = parameter_space.sample_random_parameters(seed=42+iteration)
            
            # 模拟评估
            objective_value = 0.7 + 0.05 * iteration + 0.1 * np.random.random()
            result = OptimizationResult(
                parameters=params,
                objective_value=objective_value,
                metrics={'AUROC': objective_value, 'AUPRC': objective_value - 0.1},
                iteration=iteration,
                timestamp=datetime.now(),
                evaluation_time=100.0 + 20 * iteration
            )
            
            # 更新历史
            history.add_result(result)
            
            # 更新高斯过程（简化版）
            if iteration == 1:
                X = np.array([list(params.values())[:10]])  # 取前10个参数
                y = np.array([objective_value])
                gp.fit(X, y)
            else:
                X_new = np.array([list(params.values())[:10]])
                y_new = np.array([objective_value])
                gp.update(X_new, y_new)
            
            # 创建状态
            optimizer_state = {
                'history': history,
                'parameter_space': parameter_space,
                'gaussian_process': gp,
                'acquisition_function': {'type': 'EI', 'xi': 0.01},
                'config': {
                    'max_iterations': 100,
                    'task_type': 'LDA',
                    'current_iteration': iteration
                }
            }
            
            # 保存检查点（每2次迭代）
            checkpoint_path = self.state_manager.create_checkpoint(
                optimizer_state, iteration, checkpoint_freq=2
            )
            
            if checkpoint_path:
                # 验证检查点
                assert self.state_manager.validate_checkpoint(checkpoint_path)
                
                # 测试恢复
                loaded_state = self.state_manager.load_state(checkpoint_path)
                loaded_history = loaded_state['history']
                
                assert loaded_history.total_iterations == iteration
                assert loaded_history.get_best_objective_value() is not None
        
        # 验证最终状态
        checkpoints = self.state_manager.list_checkpoints()
        valid_checkpoints = [cp for cp in checkpoints if cp['is_valid']]
        
        # 应该有3个检查点（迭代1, 2, 4）
        assert len(valid_checkpoints) == 3
        
        # 获取最新检查点并验证
        latest_checkpoint = self.state_manager.get_latest_checkpoint()
        final_state = self.state_manager.load_state(latest_checkpoint)
        
        # 最后一个检查点是迭代4，所以历史记录应该有4次迭代
        assert final_state['history'].total_iterations == 4
        assert final_state['gaussian_process'].n_observations == 4
        assert final_state['config']['current_iteration'] == 4  # 最后一个检查点是迭代4


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])