"""
采集函数实现

提供多种采集函数的实现，用于贝叶斯优化中的参数选择。
支持Expected Improvement (EI)、Probability of Improvement (PI)、
Upper Confidence Bound (UCB)和Entropy Search (ES)等采集函数。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List, Union
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution
import warnings
from datetime import datetime


class AcquisitionFunction(ABC):
    """
    采集函数基类
    
    定义了所有采集函数的通用接口和基本功能。
    子类需要实现具体的采集函数计算逻辑。
    """
    
    def __init__(self, function_type: str, **kwargs):
        """
        初始化采集函数
        
        Args:
            function_type: 采集函数类型
            **kwargs: 采集函数特定参数
        """
        self.function_type = function_type
        self.params = kwargs
        self.creation_time = datetime.now()
        
    @abstractmethod
    def evaluate(self, X: np.ndarray, gp_model, best_value: float) -> np.ndarray:
        """
        计算采集函数值
        
        Args:
            X: 候选点，形状为 (n_samples, n_features)
            gp_model: 高斯过程模型
            best_value: 当前最佳观测值
            
        Returns:
            采集函数值，形状为 (n_samples,)
        """
        pass
    
    def optimize_acquisition(self, gp_model, bounds: List[Tuple[float, float]], 
                           best_value: float, n_restarts: int = 10,
                           method: str = 'L-BFGS-B') -> Tuple[np.ndarray, float]:
        """
        优化采集函数以找到下一个评估点
        
        Args:
            gp_model: 高斯过程模型
            bounds: 参数边界，形状为 [(low1, high1), (low2, high2), ...]
            best_value: 当前最佳观测值
            n_restarts: 优化重启次数
            method: 优化方法
            
        Returns:
            (best_x, best_acquisition_value): 最佳点和对应的采集函数值
            
        Raises:
            ValueError: 当边界格式不正确时
            RuntimeError: 当优化失败时
        """
        if not bounds:
            raise ValueError("参数边界不能为空")
        
        if not all(len(bound) == 2 and bound[0] <= bound[1] for bound in bounds):
            raise ValueError("边界格式不正确，应为 [(low, high), ...]")
        
        n_dims = len(bounds)
        best_x = None
        best_acquisition_value = -np.inf
        
        # 定义目标函数（最大化采集函数 = 最小化负采集函数）
        def objective(x):
            x_reshaped = x.reshape(1, -1)
            try:
                acquisition_value = self.evaluate(x_reshaped, gp_model, best_value)[0]
                return -acquisition_value  # 最小化负值
            except Exception as e:
                warnings.warn(f"采集函数评估失败: {e}")
                return np.inf
        
        # 多次随机重启优化
        for _ in range(n_restarts):
            # 随机初始点
            x0 = np.array([
                np.random.uniform(low, high) 
                for low, high in bounds
            ])
            
            try:
                if method == 'differential_evolution':
                    # 使用差分进化算法
                    result = differential_evolution(
                        objective, 
                        bounds, 
                        seed=np.random.randint(0, 10000),
                        maxiter=100,
                        atol=1e-6,
                        tol=1e-6
                    )
                else:
                    # 使用梯度优化方法
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-6}
                    )
                
                if result.success and -result.fun > best_acquisition_value:
                    best_x = result.x
                    best_acquisition_value = -result.fun
                    
            except Exception as e:
                warnings.warn(f"优化重启失败: {e}")
                continue
        
        if best_x is None:
            raise RuntimeError("采集函数优化失败，所有重启都失败了")
        
        return best_x, best_acquisition_value
    
    def validate_parameters(self) -> bool:
        """
        验证采集函数参数的有效性
        
        Returns:
            参数是否有效
        """
        return True  # 基类默认返回True，子类可以重写
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取采集函数信息
        
        Returns:
            包含采集函数类型和参数的字典
        """
        return {
            'function_type': self.function_type,
            'parameters': self.params.copy(),
            'creation_time': self.creation_time.isoformat()
        }
    
    def update_parameters(self, **kwargs) -> None:
        """
        更新采集函数参数
        
        Args:
            **kwargs: 新的参数值
        """
        self.params.update(kwargs)


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement (EI) 采集函数
    
    平衡探索和利用，是最常用的采集函数之一。
    通过计算期望改进量来选择下一个评估点。
    """
    
    def __init__(self, xi: float = 0.01):
        """
        初始化EI采集函数
        
        Args:
            xi: 探索参数，控制探索和利用的平衡
                xi=0时纯利用，xi越大越倾向于探索
        """
        super().__init__('EI', xi=xi)
        self.xi = xi
    
    def evaluate(self, X: np.ndarray, gp_model, best_value: float) -> np.ndarray:
        """
        计算Expected Improvement值
        
        Args:
            X: 候选点，形状为 (n_samples, n_features)
            gp_model: 高斯过程模型
            best_value: 当前最佳观测值
            
        Returns:
            EI值，形状为 (n_samples,)
        """
        # 获取预测均值和标准差
        mean, std = gp_model.predict(X, return_std=True)
        
        # 避免除零错误
        std = np.maximum(std, 1e-9)
        
        # 计算改进量
        improvement = mean - best_value - self.xi
        z = improvement / std
        
        # 计算Expected Improvement
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        # 确保非负
        return np.maximum(ei, 0.0)
    
    def validate_parameters(self) -> bool:
        """验证EI参数"""
        return self.xi >= 0.0
    
    def update_xi(self, xi: float) -> None:
        """更新探索参数"""
        if xi < 0:
            raise ValueError("xi必须非负")
        self.xi = xi
        self.params['xi'] = xi


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Probability of Improvement (PI) 采集函数
    
    保守的改进策略，计算改进的概率。
    相比EI更加保守，倾向于选择改进概率高的点。
    """
    
    def __init__(self, xi: float = 0.01):
        """
        初始化PI采集函数
        
        Args:
            xi: 探索参数，控制探索和利用的平衡
        """
        super().__init__('PI', xi=xi)
        self.xi = xi
    
    def evaluate(self, X: np.ndarray, gp_model, best_value: float) -> np.ndarray:
        """
        计算Probability of Improvement值
        
        Args:
            X: 候选点，形状为 (n_samples, n_features)
            gp_model: 高斯过程模型
            best_value: 当前最佳观测值
            
        Returns:
            PI值，形状为 (n_samples,)
        """
        # 获取预测均值和标准差
        mean, std = gp_model.predict(X, return_std=True)
        
        # 避免除零错误
        std = np.maximum(std, 1e-9)
        
        # 计算改进量
        improvement = mean - best_value - self.xi
        z = improvement / std
        
        # 计算改进概率
        return norm.cdf(z)
    
    def validate_parameters(self) -> bool:
        """验证PI参数"""
        return self.xi >= 0.0
    
    def update_xi(self, xi: float) -> None:
        """更新探索参数"""
        if xi < 0:
            raise ValueError("xi必须非负")
        self.xi = xi
        self.params['xi'] = xi


class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound (UCB) 采集函数
    
    基于置信区间的选择策略，选择预测均值加上置信区间的点。
    通过kappa参数控制探索程度。
    """
    
    def __init__(self, kappa: float = 2.576):
        """
        初始化UCB采集函数
        
        Args:
            kappa: 置信区间参数，控制探索程度
                   kappa=2.576对应99%置信区间
        """
        super().__init__('UCB', kappa=kappa)
        self.kappa = kappa
    
    def evaluate(self, X: np.ndarray, gp_model, best_value: float) -> np.ndarray:
        """
        计算Upper Confidence Bound值
        
        Args:
            X: 候选点，形状为 (n_samples, n_features)
            gp_model: 高斯过程模型
            best_value: 当前最佳观测值（UCB中不直接使用）
            
        Returns:
            UCB值，形状为 (n_samples,)
        """
        # 获取预测均值和标准差
        mean, std = gp_model.predict(X, return_std=True)
        
        # 计算上置信界
        return mean + self.kappa * std
    
    def validate_parameters(self) -> bool:
        """验证UCB参数"""
        return self.kappa > 0.0
    
    def update_kappa(self, kappa: float) -> None:
        """更新置信区间参数"""
        if kappa <= 0:
            raise ValueError("kappa必须为正数")
        self.kappa = kappa
        self.params['kappa'] = kappa


class EntropySearch(AcquisitionFunction):
    """
    Entropy Search (ES) 采集函数
    
    基于信息熵的选择策略，选择能最大化信息增益的点。
    这是一个更高级的采集函数，计算复杂度较高但效果通常更好。
    """
    
    def __init__(self, n_samples: int = 100, temperature: float = 1.0):
        """
        初始化ES采集函数
        
        Args:
            n_samples: 蒙特卡洛采样数量
            temperature: 温度参数，控制采样的随机性
        """
        super().__init__('ES', n_samples=n_samples, temperature=temperature)
        self.n_samples = n_samples
        self.temperature = temperature
    
    def evaluate(self, X: np.ndarray, gp_model, best_value: float) -> np.ndarray:
        """
        计算Entropy Search值
        
        Args:
            X: 候选点，形状为 (n_samples, n_features)
            gp_model: 高斯过程模型
            best_value: 当前最佳观测值
            
        Returns:
            ES值，形状为 (n_samples,)
        """
        # 获取预测均值和协方差
        mean, cov = gp_model.predict(X, return_cov=True)
        
        if cov.ndim == 1:
            # 如果返回的是方差而不是协方差矩阵
            std = np.sqrt(cov)
            cov = np.diag(cov)
        else:
            std = np.sqrt(np.diag(cov))
        
        # 简化的熵搜索实现
        # 真正的ES需要复杂的蒙特卡洛积分，这里使用近似方法
        
        # 计算预测不确定性（熵的代理）
        entropy_proxy = np.log(2 * np.pi * np.e * std**2) / 2
        
        # 考虑改进的可能性
        improvement_prob = norm.cdf((mean - best_value) / (std + 1e-9))
        
        # 结合不确定性和改进概率
        es_values = entropy_proxy * improvement_prob
        
        return es_values
    
    def validate_parameters(self) -> bool:
        """验证ES参数"""
        return self.n_samples > 0 and self.temperature > 0.0
    
    def update_n_samples(self, n_samples: int) -> None:
        """更新采样数量"""
        if n_samples <= 0:
            raise ValueError("n_samples必须为正整数")
        self.n_samples = n_samples
        self.params['n_samples'] = n_samples
    
    def update_temperature(self, temperature: float) -> None:
        """更新温度参数"""
        if temperature <= 0:
            raise ValueError("temperature必须为正数")
        self.temperature = temperature
        self.params['temperature'] = temperature


def create_acquisition_function(function_type: str, **kwargs) -> AcquisitionFunction:
    """
    创建采集函数实例的工厂函数
    
    Args:
        function_type: 采集函数类型 ('EI', 'PI', 'UCB', 'ES')
        **kwargs: 采集函数特定参数
        
    Returns:
        采集函数实例
        
    Raises:
        ValueError: 当采集函数类型不支持时
    """
    function_type = function_type.upper()
    
    if function_type == 'EI':
        return ExpectedImprovement(**kwargs)
    elif function_type == 'PI':
        return ProbabilityOfImprovement(**kwargs)
    elif function_type == 'UCB':
        return UpperConfidenceBound(**kwargs)
    elif function_type == 'ES':
        return EntropySearch(**kwargs)
    else:
        raise ValueError(f"不支持的采集函数类型: {function_type}")


class AcquisitionOptimizer:
    """
    采集函数优化器
    
    提供多种优化策略来找到采集函数的最优点。
    支持梯度优化、差分进化和网格搜索等方法。
    """
    
    def __init__(self, method: str = 'L-BFGS-B', n_restarts: int = 10):
        """
        初始化优化器
        
        Args:
            method: 优化方法 ('L-BFGS-B', 'differential_evolution', 'grid_search')
            n_restarts: 随机重启次数
        """
        self.method = method
        self.n_restarts = n_restarts
        
        # 支持的优化方法
        self.supported_methods = ['L-BFGS-B', 'differential_evolution', 'grid_search']
        
        if method not in self.supported_methods:
            raise ValueError(f"不支持的优化方法: {method}")
    
    def optimize(self, acquisition_func: AcquisitionFunction, 
                gp_model, bounds: List[Tuple[float, float]], 
                best_value: float) -> Tuple[np.ndarray, float]:
        """
        优化采集函数
        
        Args:
            acquisition_func: 采集函数实例
            gp_model: 高斯过程模型
            bounds: 参数边界
            best_value: 当前最佳观测值
            
        Returns:
            (best_x, best_acquisition_value): 最佳点和对应的采集函数值
        """
        if self.method == 'grid_search':
            return self._grid_search_optimize(acquisition_func, gp_model, bounds, best_value)
        else:
            return acquisition_func.optimize_acquisition(
                gp_model, bounds, best_value, 
                n_restarts=self.n_restarts, method=self.method
            )
    
    def _grid_search_optimize(self, acquisition_func: AcquisitionFunction,
                            gp_model, bounds: List[Tuple[float, float]],
                            best_value: float, n_points_per_dim: int = 20) -> Tuple[np.ndarray, float]:
        """
        网格搜索优化
        
        Args:
            acquisition_func: 采集函数实例
            gp_model: 高斯过程模型
            bounds: 参数边界
            best_value: 当前最佳观测值
            n_points_per_dim: 每个维度的网格点数
            
        Returns:
            (best_x, best_acquisition_value): 最佳点和对应的采集函数值
        """
        # 创建网格
        grid_axes = []
        for low, high in bounds:
            grid_axes.append(np.linspace(low, high, n_points_per_dim))
        
        # 生成所有网格点
        grid_points = np.array(np.meshgrid(*grid_axes)).T.reshape(-1, len(bounds))
        
        # 计算所有网格点的采集函数值
        acquisition_values = acquisition_func.evaluate(grid_points, gp_model, best_value)
        
        # 找到最佳点
        best_idx = np.argmax(acquisition_values)
        best_x = grid_points[best_idx]
        best_acquisition_value = acquisition_values[best_idx]
        
        return best_x, best_acquisition_value


if __name__ == "__main__":
    # 测试代码
    print("测试采集函数实现...")
    
    # 创建模拟的高斯过程模型
    class MockGP:
        def predict(self, X, return_std=True, return_cov=False):
            mean = np.sum(X**2, axis=1)
            if return_cov:
                std = np.ones(len(X)) * 0.1
                cov = np.diag(std**2)
                return mean, cov
            elif return_std:
                std = np.ones(len(X)) * 0.1
                return mean, std
            else:
                return mean
    
    mock_gp = MockGP()
    
    # 测试数据
    X_test = np.array([[0.5, 0.5], [1.0, 1.0], [-0.5, -0.5]])
    best_value = 1.0
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    
    # 测试所有采集函数
    acquisition_functions = [
        ('EI', {'xi': 0.01}),
        ('PI', {'xi': 0.01}),
        ('UCB', {'kappa': 2.576}),
        ('ES', {'n_samples': 50})
    ]
    
    for func_type, params in acquisition_functions:
        print(f"\n测试 {func_type} 采集函数:")
        
        # 创建采集函数
        acq_func = create_acquisition_function(func_type, **params)
        print(f"  创建成功: {acq_func.get_info()}")
        
        # 测试参数验证
        is_valid = acq_func.validate_parameters()
        print(f"  参数验证: {is_valid}")
        
        # 测试评估
        try:
            values = acq_func.evaluate(X_test, mock_gp, best_value)
            print(f"  采集函数值: {values}")
        except Exception as e:
            print(f"  评估失败: {e}")
        
        # 测试优化
        try:
            optimizer = AcquisitionOptimizer(method='L-BFGS-B', n_restarts=3)
            best_x, best_val = optimizer.optimize(acq_func, mock_gp, bounds, best_value)
            print(f"  优化结果: x={best_x}, value={best_val:.6f}")
        except Exception as e:
            print(f"  优化失败: {e}")
    
    # 测试网格搜索优化
    print(f"\n测试网格搜索优化:")
    try:
        grid_optimizer = AcquisitionOptimizer(method='grid_search')
        ei_func = create_acquisition_function('EI', xi=0.01)
        best_x, best_val = grid_optimizer.optimize(ei_func, mock_gp, bounds, best_value)
        print(f"  网格搜索结果: x={best_x}, value={best_val:.6f}")
    except Exception as e:
        print(f"  网格搜索失败: {e}")
    
    print("\n采集函数测试完成!")