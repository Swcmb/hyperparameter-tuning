"""
高斯过程模型实现

基于scikit-learn的高斯过程包装类，用于贝叶斯优化中的代理模型。
支持模型拟合、预测和增量更新功能，使用Matérn 5/2核函数处理非光滑目标函数。
"""

import numpy as np
import pickle
from typing import Tuple, Dict, Any, Optional, List
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import warnings
from datetime import datetime


class GaussianProcess:
    """
    高斯过程模型包装类
    
    基于scikit-learn的GaussianProcessRegressor，专门为贝叶斯优化设计。
    使用Matérn 5/2核函数，适合处理非光滑的目标函数。
    """
    
    def __init__(self, 
                 length_scale: float = 1.0,
                 length_scale_bounds: Tuple[float, float] = (1e-5, 1e5),
                 noise_level: float = 1e-6,
                 noise_level_bounds: Tuple[float, float] = (1e-10, 1e-1),
                 constant_value: float = 1.0,
                 constant_value_bounds: Tuple[float, float] = (1e-5, 1e5),
                 n_restarts_optimizer: int = 10,
                 random_state: Optional[int] = None):
        """
        初始化高斯过程模型
        
        Args:
            length_scale: Matérn核的长度尺度参数
            length_scale_bounds: 长度尺度的优化边界
            noise_level: 观测噪声水平
            noise_level_bounds: 噪声水平的优化边界
            constant_value: 常数核的值
            constant_value_bounds: 常数核值的优化边界
            n_restarts_optimizer: 超参数优化的重启次数
            random_state: 随机种子
        """
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        
        # 创建核函数：ConstantKernel * Matérn + WhiteKernel
        # Matérn 5/2核适合非光滑函数，WhiteKernel处理噪声
        self.kernel = (
            ConstantKernel(constant_value, constant_value_bounds) *
            Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=2.5) +
            WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
        )
        
        # 创建高斯过程回归器
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
            normalize_y=True,  # 标准化目标值，提高数值稳定性
            alpha=1e-10  # 对角线正则化项，防止数值问题
        )
        
        # 存储训练数据
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        
        # 模型统计信息
        self.n_observations: int = 0
        self.last_fit_time: Optional[datetime] = None
        self.log_marginal_likelihood: Optional[float] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcess':
        """
        拟合高斯过程模型
        
        Args:
            X: 输入特征矩阵，形状为 (n_samples, n_features)
            y: 目标值向量，形状为 (n_samples,)
            
        Returns:
            self: 返回自身以支持链式调用
            
        Raises:
            ValueError: 当输入数据格式不正确时
        """
        # 输入验证
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError(f"X必须是二维数组，当前维度: {X.ndim}")
        
        if y.ndim != 1:
            raise ValueError(f"y必须是一维数组，当前维度: {y.ndim}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X和y的样本数不匹配: {X.shape[0]} vs {y.shape[0]}")
        
        if X.shape[0] == 0:
            raise ValueError("训练数据不能为空")
        
        # 检查数据中的无效值
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X中包含NaN或无穷大值")
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y中包含NaN或无穷大值")
        
        try:
            # 拟合模型
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.gp.fit(X, y)
            
            # 存储训练数据
            self.X_train = X.copy()
            self.y_train = y.copy()
            self.is_fitted = True
            self.n_observations = X.shape[0]
            self.last_fit_time = datetime.now()
            
            # 记录模型质量指标
            self.log_marginal_likelihood = self.gp.log_marginal_likelihood()
            
        except Exception as e:
            raise RuntimeError(f"高斯过程拟合失败: {str(e)}")
        
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = True, 
                return_cov: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用高斯过程进行预测
        
        Args:
            X: 预测点的输入特征，形状为 (n_samples, n_features)
            return_std: 是否返回预测标准差
            return_cov: 是否返回协方差矩阵（优先级高于return_std）
            
        Returns:
            mean: 预测均值，形状为 (n_samples,)
            std_or_cov: 预测标准差或协方差矩阵
            
        Raises:
            RuntimeError: 当模型未拟合时
            ValueError: 当输入数据格式不正确时
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用fit()方法")
        
        # 输入验证
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X必须是二维数组，当前维度: {X.ndim}")
        
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(f"特征维度不匹配: 期望{self.X_train.shape[1]}，实际{X.shape[1]}")
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X中包含NaN或无穷大值")
        
        try:
            # 进行预测
            if return_cov:
                mean, cov = self.gp.predict(X, return_cov=True)
                return mean, cov
            elif return_std:
                mean, std = self.gp.predict(X, return_std=True)
                return mean, std
            else:
                mean = self.gp.predict(X, return_std=False)
                return mean, np.zeros_like(mean)
                
        except Exception as e:
            raise RuntimeError(f"高斯过程预测失败: {str(e)}")
    
    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> 'GaussianProcess':
        """
        增量更新高斯过程模型
        
        将新的观测数据添加到现有训练集中，并重新拟合模型。
        
        Args:
            X_new: 新的输入特征，形状为 (n_new_samples, n_features)
            y_new: 新的目标值，形状为 (n_new_samples,)
            
        Returns:
            self: 返回自身以支持链式调用
            
        Raises:
            RuntimeError: 当模型未初始拟合时
            ValueError: 当输入数据格式不正确时
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未初始拟合，请先调用fit()方法")
        
        # 输入验证
        X_new = np.asarray(X_new)
        y_new = np.asarray(y_new)
        
        if X_new.ndim != 2:
            raise ValueError(f"X_new必须是二维数组，当前维度: {X_new.ndim}")
        
        if y_new.ndim != 1:
            raise ValueError(f"y_new必须是一维数组，当前维度: {y_new.ndim}")
        
        if X_new.shape[0] != y_new.shape[0]:
            raise ValueError(f"X_new和y_new的样本数不匹配: {X_new.shape[0]} vs {y_new.shape[0]}")
        
        if X_new.shape[1] != self.X_train.shape[1]:
            raise ValueError(f"特征维度不匹配: 期望{self.X_train.shape[1]}，实际{X_new.shape[1]}")
        
        # 合并训练数据
        X_combined = np.vstack([self.X_train, X_new])
        y_combined = np.hstack([self.y_train, y_new])
        
        # 重新拟合模型
        return self.fit(X_combined, y_combined)
    
    def get_hyperparameters(self) -> Dict[str, float]:
        """
        获取当前的核函数超参数
        
        Returns:
            超参数字典，包含长度尺度、噪声水平等
            
        Raises:
            RuntimeError: 当模型未拟合时
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，无法获取超参数")
        
        # 解析核函数参数
        kernel_params = self.gp.kernel_.get_params()
        
        hyperparams = {}
        
        # 提取主要超参数
        for key, value in kernel_params.items():
            if isinstance(value, (int, float, np.number)):
                hyperparams[key] = float(value)
        
        # 添加模型质量指标
        hyperparams['log_marginal_likelihood'] = self.log_marginal_likelihood
        hyperparams['n_observations'] = self.n_observations
        
        return hyperparams
    
    def compute_acquisition_values(self, X: np.ndarray, 
                                  acquisition_type: str = 'EI',
                                  xi: float = 0.01,
                                  kappa: float = 2.576) -> np.ndarray:
        """
        计算采集函数值
        
        Args:
            X: 候选点，形状为 (n_samples, n_features)
            acquisition_type: 采集函数类型 ('EI', 'PI', 'UCB')
            xi: EI和PI的探索参数
            kappa: UCB的探索参数
            
        Returns:
            采集函数值，形状为 (n_samples,)
            
        Raises:
            RuntimeError: 当模型未拟合时
            ValueError: 当采集函数类型不支持时
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，无法计算采集函数")
        
        if acquisition_type not in ['EI', 'PI', 'UCB']:
            raise ValueError(f"不支持的采集函数类型: {acquisition_type}")
        
        # 获取预测均值和标准差
        mean, std = self.predict(X, return_std=True)
        
        # 当前最佳观测值
        best_value = np.max(self.y_train)
        
        if acquisition_type == 'EI':
            # Expected Improvement
            return self._expected_improvement(mean, std, best_value, xi)
        
        elif acquisition_type == 'PI':
            # Probability of Improvement
            return self._probability_of_improvement(mean, std, best_value, xi)
        
        elif acquisition_type == 'UCB':
            # Upper Confidence Bound
            return self._upper_confidence_bound(mean, std, kappa)
    
    def _expected_improvement(self, mean: np.ndarray, std: np.ndarray, 
                             best_value: float, xi: float) -> np.ndarray:
        """计算Expected Improvement采集函数"""
        from scipy.stats import norm
        
        # 避免除零错误
        std = np.maximum(std, 1e-9)
        
        improvement = mean - best_value - xi
        z = improvement / std
        
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        # 确保非负
        return np.maximum(ei, 0.0)
    
    def _probability_of_improvement(self, mean: np.ndarray, std: np.ndarray,
                                   best_value: float, xi: float) -> np.ndarray:
        """计算Probability of Improvement采集函数"""
        from scipy.stats import norm
        
        # 避免除零错误
        std = np.maximum(std, 1e-9)
        
        improvement = mean - best_value - xi
        z = improvement / std
        
        return norm.cdf(z)
    
    def _upper_confidence_bound(self, mean: np.ndarray, std: np.ndarray,
                               kappa: float) -> np.ndarray:
        """计算Upper Confidence Bound采集函数"""
        return mean + kappa * std
    
    def get_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取当前的训练数据
        
        Returns:
            (X_train, y_train): 训练特征和目标值的副本
        """
        if self.is_fitted:
            return self.X_train.copy(), self.y_train.copy()
        else:
            return None, None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型的详细信息
        
        Returns:
            包含模型状态和统计信息的字典
        """
        info = {
            'is_fitted': self.is_fitted,
            'n_observations': self.n_observations,
            'last_fit_time': self.last_fit_time.isoformat() if self.last_fit_time else None,
            'log_marginal_likelihood': self.log_marginal_likelihood,
            'kernel_type': str(self.kernel),
            'n_restarts_optimizer': self.n_restarts_optimizer,
            'random_state': self.random_state
        }
        
        if self.is_fitted:
            info['training_data_shape'] = self.X_train.shape
            info['hyperparameters'] = self.get_hyperparameters()
        
        return info
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
            
        Raises:
            RuntimeError: 当保存失败时
        """
        try:
            model_data = {
                'gp': self.gp,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'is_fitted': self.is_fitted,
                'n_observations': self.n_observations,
                'last_fit_time': self.last_fit_time,
                'log_marginal_likelihood': self.log_marginal_likelihood,
                'init_params': {
                    'length_scale': self.length_scale,
                    'length_scale_bounds': self.length_scale_bounds,
                    'noise_level': self.noise_level,
                    'noise_level_bounds': self.noise_level_bounds,
                    'constant_value': self.constant_value,
                    'constant_value_bounds': self.constant_value_bounds,
                    'n_restarts_optimizer': self.n_restarts_optimizer,
                    'random_state': self.random_state
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            raise RuntimeError(f"保存模型失败: {str(e)}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'GaussianProcess':
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            加载的GaussianProcess实例
            
        Raises:
            RuntimeError: 当加载失败时
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # 重建模型实例
            init_params = model_data['init_params']
            model = cls(**init_params)
            
            # 恢复状态
            model.gp = model_data['gp']
            model.X_train = model_data['X_train']
            model.y_train = model_data['y_train']
            model.is_fitted = model_data['is_fitted']
            model.n_observations = model_data['n_observations']
            model.last_fit_time = model_data['last_fit_time']
            model.log_marginal_likelihood = model_data['log_marginal_likelihood']
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")
    
    def reset(self) -> None:
        """
        重置模型状态
        
        清除所有训练数据和拟合状态，但保留初始化参数
        """
        # 重新创建核函数和GP实例
        self.kernel = (
            ConstantKernel(self.constant_value, self.constant_value_bounds) *
            Matern(length_scale=self.length_scale, length_scale_bounds=self.length_scale_bounds, nu=2.5) +
            WhiteKernel(noise_level=self.noise_level, noise_level_bounds=self.noise_level_bounds)
        )
        
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            normalize_y=True,
            alpha=1e-10
        )
        
        # 清除状态
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        self.n_observations = 0
        self.last_fit_time = None
        self.log_marginal_likelihood = None


def create_default_gaussian_process(random_state: Optional[int] = None) -> GaussianProcess:
    """
    创建默认配置的高斯过程模型
    
    使用适合贝叶斯优化的默认参数配置
    
    Args:
        random_state: 随机种子
        
    Returns:
        配置好的GaussianProcess实例
    """
    return GaussianProcess(
        length_scale=1.0,
        length_scale_bounds=(1e-3, 1e3),
        noise_level=1e-6,
        noise_level_bounds=(1e-10, 1e-1),
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3),
        n_restarts_optimizer=10,
        random_state=random_state
    )


if __name__ == "__main__":
    # 测试代码
    print("测试高斯过程模型...")
    
    # 创建测试数据
    np.random.seed(42)
    X_train = np.random.uniform(-3, 3, (10, 2))
    y_train = np.sum(X_train**2, axis=1) + 0.1 * np.random.randn(10)
    
    X_test = np.random.uniform(-3, 3, (5, 2))
    
    # 测试模型创建和拟合
    gp = create_default_gaussian_process(random_state=42)
    print(f"创建高斯过程模型: {gp.get_model_info()}")
    
    # 测试拟合
    gp.fit(X_train, y_train)
    print(f"拟合完成，观测数量: {gp.n_observations}")
    print(f"对数边际似然: {gp.log_marginal_likelihood:.4f}")
    
    # 测试预测
    mean, std = gp.predict(X_test)
    print(f"预测结果:")
    print(f"  均值: {mean}")
    print(f"  标准差: {std}")
    
    # 测试超参数
    hyperparams = gp.get_hyperparameters()
    print(f"超参数: {hyperparams}")
    
    # 测试采集函数
    ei_values = gp.compute_acquisition_values(X_test, 'EI')
    pi_values = gp.compute_acquisition_values(X_test, 'PI')
    ucb_values = gp.compute_acquisition_values(X_test, 'UCB')
    
    print(f"采集函数值:")
    print(f"  EI: {ei_values}")
    print(f"  PI: {pi_values}")
    print(f"  UCB: {ucb_values}")
    
    # 测试增量更新
    X_new = np.random.uniform(-3, 3, (3, 2))
    y_new = np.sum(X_new**2, axis=1) + 0.1 * np.random.randn(3)
    
    gp.update(X_new, y_new)
    print(f"增量更新后观测数量: {gp.n_observations}")
    
    # 测试模型保存和加载
    try:
        gp.save_model('test_gp_model.pkl')
        loaded_gp = GaussianProcess.load_model('test_gp_model.pkl')
        print(f"模型保存和加载测试成功")
        print(f"加载后观测数量: {loaded_gp.n_observations}")
        
        # 清理测试文件
        import os
        os.remove('test_gp_model.pkl')
        
    except Exception as e:
        print(f"模型保存/加载测试失败: {e}")
    
    print("高斯过程模型测试完成!")