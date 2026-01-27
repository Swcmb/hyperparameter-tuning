"""
高斯过程模型使用示例

展示如何使用高斯过程模型进行贝叶斯优化
"""

import numpy as np
import matplotlib.pyplot as plt
from gaussian_process import create_default_gaussian_process
from autodl_core import create_default_parameter_space


def demo_1d_optimization():
    """演示一维函数优化"""
    print("=== 一维函数优化演示 ===")
    
    # 定义目标函数（带噪声的正弦函数）
    def objective_function(x):
        return -(x - 0.3)**2 * np.sin(5*x) + 0.1 * np.random.randn()
    
    # 创建高斯过程模型
    gp = create_default_gaussian_process(random_state=42)
    
    # 初始观测点
    X_init = np.array([[0.1], [0.9]])
    y_init = np.array([objective_function(x[0]) for x in X_init])
    
    # 拟合初始模型
    gp.fit(X_init, y_init)
    print(f"初始观测点数量: {gp.n_observations}")
    
    # 贝叶斯优化迭代
    X_all = X_init.copy()
    y_all = y_init.copy()
    
    for iteration in range(8):
        # 生成候选点
        X_candidates = np.linspace(0, 1, 100).reshape(-1, 1)
        
        # 计算采集函数值
        ei_values = gp.compute_acquisition_values(X_candidates, 'EI')
        
        # 选择最佳候选点
        best_idx = np.argmax(ei_values)
        x_next = X_candidates[best_idx]
        y_next = objective_function(x_next[0])
        
        print(f"迭代 {iteration+1}: x={x_next[0]:.3f}, y={y_next:.3f}, EI={ei_values[best_idx]:.6f}")
        
        # 更新模型
        gp.update(x_next.reshape(1, -1), np.array([y_next]))
        
        # 记录历史
        X_all = np.vstack([X_all, x_next])
        y_all = np.append(y_all, y_next)
    
    # 找到最佳点
    best_idx = np.argmax(y_all)
    print(f"最佳点: x={X_all[best_idx][0]:.3f}, y={y_all[best_idx]:.3f}")
    
    return X_all, y_all, gp


def demo_parameter_space_optimization():
    """演示参数空间优化"""
    print("\n=== 参数空间优化演示 ===")
    
    # 创建参数空间
    param_space = create_default_parameter_space()
    continuous_params = param_space.get_continuous_parameter_names()
    print(f"连续参数数量: {len(continuous_params)}")
    print(f"连续参数: {continuous_params[:5]}...")  # 显示前5个
    
    # 创建高斯过程模型
    gp = create_default_gaussian_process(random_state=42)
    
    # 模拟目标函数（基于学习率和dropout的简单函数）
    def mock_auroc(params):
        lr = float(params['lr'])
        dropout = float(params['dropout'])
        alpha = float(params['alpha'])
        
        # 模拟AUROC分数
        lr_score = -0.1 * (np.log10(lr) + 3)**2  # lr在1e-3附近最优
        dropout_score = -0.2 * (dropout - 0.3)**2  # dropout在0.3附近最优
        alpha_score = 0.05 * alpha
        noise = np.random.RandomState(hash(str(params)) % 2**32).normal(0, 0.02)
        
        base_score = 0.75
        total_score = base_score + lr_score + dropout_score + alpha_score + noise
        return max(0.5, min(0.95, total_score))
    
    # 转换参数到数组的辅助函数
    def params_to_array(params_list):
        arrays = []
        for params in params_list:
            row = []
            for name in continuous_params:
                value = params[name]
                if name in ['lr', 'weight_decay'] and value > 0:
                    row.append(np.log(value))
                else:
                    row.append(float(value))
            arrays.append(row)
        return np.array(arrays)
    
    # 初始随机采样
    initial_params = []
    initial_scores = []
    
    for i in range(5):
        params = param_space.sample_random_parameters(seed=42+i)
        score = mock_auroc(params)
        initial_params.append(params)
        initial_scores.append(score)
        print(f"初始点 {i+1}: AUROC={score:.4f}, lr={params['lr']:.2e}, dropout={params['dropout']:.3f}")
    
    # 训练初始模型
    X_init = params_to_array(initial_params)
    y_init = np.array(initial_scores)
    gp.fit(X_init, y_init)
    
    # 贝叶斯优化迭代
    all_params = initial_params.copy()
    all_scores = initial_scores.copy()
    
    for iteration in range(10):
        # 生成候选参数
        candidates = []
        for i in range(50):
            candidate = param_space.sample_random_parameters(seed=1000+iteration*50+i)
            candidates.append(candidate)
        
        X_candidates = params_to_array(candidates)
        
        # 计算采集函数值
        ei_values = gp.compute_acquisition_values(X_candidates, 'EI')
        
        # 选择最佳候选
        best_idx = np.argmax(ei_values)
        best_params = candidates[best_idx]
        best_score = mock_auroc(best_params)
        
        print(f"迭代 {iteration+1}: AUROC={best_score:.4f}, lr={best_params['lr']:.2e}, "
              f"dropout={best_params['dropout']:.3f}, EI={ei_values[best_idx]:.6f}")
        
        # 更新模型
        X_new = params_to_array([best_params])
        gp.update(X_new, np.array([best_score]))
        
        # 记录历史
        all_params.append(best_params)
        all_scores.append(best_score)
    
    # 找到最佳参数
    best_idx = np.argmax(all_scores)
    best_params = all_params[best_idx]
    best_score = all_scores[best_idx]
    
    print(f"\n最佳参数组合:")
    print(f"  AUROC: {best_score:.4f}")
    print(f"  lr: {best_params['lr']:.2e}")
    print(f"  dropout: {best_params['dropout']:.3f}")
    print(f"  alpha: {best_params['alpha']:.3f}")
    print(f"  fusion_strategy: {best_params['fusion_strategy']}")
    
    return all_params, all_scores, gp


def demo_acquisition_functions():
    """演示不同采集函数的行为"""
    print("\n=== 采集函数比较演示 ===")
    
    # 创建简单的1D测试数据
    X_train = np.array([[0.2], [0.7]])
    y_train = np.array([0.3, 0.8])
    
    gp = create_default_gaussian_process(random_state=42)
    gp.fit(X_train, y_train)
    
    # 测试点
    X_test = np.linspace(0, 1, 50).reshape(-1, 1)
    
    # 计算不同采集函数
    ei_values = gp.compute_acquisition_values(X_test, 'EI', xi=0.01)
    pi_values = gp.compute_acquisition_values(X_test, 'PI', xi=0.01)
    ucb_values = gp.compute_acquisition_values(X_test, 'UCB', kappa=2.576)
    
    # 获取预测
    mean, std = gp.predict(X_test)
    
    print("采集函数统计:")
    print(f"  EI - 最大值: {np.max(ei_values):.6f}, 最大值位置: {X_test[np.argmax(ei_values)][0]:.3f}")
    print(f"  PI - 最大值: {np.max(pi_values):.6f}, 最大值位置: {X_test[np.argmax(pi_values)][0]:.3f}")
    print(f"  UCB - 最大值: {np.max(ucb_values):.6f}, 最大值位置: {X_test[np.argmax(ucb_values)][0]:.3f}")
    
    # 显示模型信息
    model_info = gp.get_model_info()
    print(f"\n模型信息:")
    print(f"  观测数量: {model_info['n_observations']}")
    print(f"  对数边际似然: {model_info['log_marginal_likelihood']:.4f}")
    
    return X_test, mean, std, ei_values, pi_values, ucb_values


def main():
    """主函数"""
    print("高斯过程模型演示")
    print("=" * 50)
    
    # 演示1: 一维优化
    X_1d, y_1d, gp_1d = demo_1d_optimization()
    
    # 演示2: 参数空间优化
    params_history, scores_history, gp_params = demo_parameter_space_optimization()
    
    # 演示3: 采集函数比较
    X_test, mean, std, ei, pi, ucb = demo_acquisition_functions()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("\n高斯过程模型主要特性:")
    print("✓ 支持Matérn 5/2核函数，适合非光滑目标函数")
    print("✓ 支持多种采集函数（EI、PI、UCB）")
    print("✓ 支持增量更新和模型持久化")
    print("✓ 与参数空间管理器无缝集成")
    print("✓ 提供详细的模型信息和超参数")


if __name__ == "__main__":
    main()