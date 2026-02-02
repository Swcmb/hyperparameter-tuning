#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对 logs/autodl_20260201_171815_20260201_171815 的详细分析脚本
生成更深入的分析图表和报告
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class DetailedAnalyzer:
    def __init__(self, log_dir):
        """初始化详细分析器"""
        self.log_dir = Path(log_dir)
        self.structured_log_file = self.log_dir / "autodl_20260201_171815_structured.jsonl"
        
        # 存储解析后的数据
        self.all_data = []
        self.iterations_data = []
        self.best_results = []
        
        # 输出目录
        self.output_dir = Path("detailed_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_and_parse_data(self):
        """加载并解析所有数据"""
        print("正在加载和解析日志数据...")
        
        with open(self.structured_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    self.all_data.append(data)
                except:
                    continue
                    
        # 提取迭代数据
        self._extract_iteration_data()
        print(f"成功解析 {len(self.iterations_data)} 次迭代数据")
        
    def _extract_iteration_data(self):
        """提取每次迭代的完整数据"""
        current_iteration = {}
        
        for entry in self.all_data:
            tag = entry.get('tag', '')
            message = entry.get('message', '')
            
            if tag == 'ITERATION' and message.isdigit():
                # 开始新的迭代
                if current_iteration:
                    self.iterations_data.append(current_iteration.copy())
                current_iteration = {
                    'iteration': int(message),
                    'timestamp': entry.get('timestamp', '')
                }
                
            elif tag == 'SUGGESTED_PARAMS' and 'structured_data' in entry:
                # 添加建议的参数
                current_iteration.update(entry['structured_data'])
                
            elif tag == 'NEW_RESULT' and 'structured_data' in entry:
                # 添加评估结果
                result_data = entry['structured_data']
                current_iteration.update({
                    'objective_value': result_data.get('objective_value', 0),
                    'evaluation_time': result_data.get('evaluation_time', 0),
                    'AUROC': result_data.get('main_metrics', {}).get('AUROC', 0),
                    'AUPRC': result_data.get('main_metrics', {}).get('AUPRC', 0),
                    'F1': result_data.get('main_metrics', {}).get('F1', 0),
                    'precision': result_data.get('main_metrics', {}).get('precision', 0),
                    'recall': result_data.get('main_metrics', {}).get('recall', 0),
                    'loss': result_data.get('main_metrics', {}).get('loss', 0)
                })
                
        # 添加最后一次迭代
        if current_iteration:
            self.iterations_data.append(current_iteration)
            
    def create_comprehensive_performance_plot(self):
        """创建综合性能分析图"""
        if not self.iterations_data:
            return
            
        df = pd.DataFrame(self.iterations_data)
        df = df.sort_values('iteration')
        
        # 创建大图表
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 主要指标趋势 (2x2 子图)
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(df['iteration'], df['AUROC'], 'b-o', linewidth=2, markersize=4, label='AUROC')
        ax1.plot(df['iteration'], df['AUPRC'], 'r-s', linewidth=2, markersize=4, label='AUPRC')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('性能指标')
        ax1.set_title('主要性能指标趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(df['iteration'], df['F1'], 'g-^', linewidth=2, markersize=4, label='F1')
        ax2.plot(df['iteration'], df['precision'], 'm-d', linewidth=2, markersize=4, label='Precision')
        ax2.plot(df['iteration'], df['recall'], 'c-v', linewidth=2, markersize=4, label='Recall')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('分类指标')
        ax2.set_title('分类性能指标趋势')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 2. 累积最佳值
        ax3 = plt.subplot(3, 3, 3)
        cumulative_best = df['AUROC'].cummax()
        ax3.plot(df['iteration'], df['AUROC'], 'lightblue', alpha=0.6, label='当前值')
        ax3.plot(df['iteration'], cumulative_best, 'darkblue', linewidth=3, label='累积最佳')
        ax3.fill_between(df['iteration'], df['AUROC'], cumulative_best, alpha=0.3)
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('AUROC')
        ax3.set_title('AUROC 累积最佳值')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 3. 评估时间分析
        ax4 = plt.subplot(3, 3, 4)
        ax4.bar(df['iteration'], df['evaluation_time'], alpha=0.7, color='orange')
        ax4.set_xlabel('迭代次数')
        ax4.set_ylabel('评估时间 (秒)')
        ax4.set_title('每次迭代评估时间')
        ax4.grid(True, alpha=0.3)
        
        # 4. 损失函数趋势
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(df['iteration'], df['loss'], 'r-o', linewidth=2, markersize=4)
        ax5.set_xlabel('迭代次数')
        ax5.set_ylabel('损失值')
        ax5.set_title('损失函数变化')
        ax5.grid(True, alpha=0.3)
        
        # 5. 性能改进分析
        ax6 = plt.subplot(3, 3, 6)
        improvements = df['AUROC'].diff().fillna(0)
        colors = ['green' if x > 0 else 'red' for x in improvements]
        ax6.bar(df['iteration'], improvements, color=colors, alpha=0.7)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.set_xlabel('迭代次数')
        ax6.set_ylabel('AUROC 改进量')
        ax6.set_title('每次迭代的性能改进')
        ax6.grid(True, alpha=0.3)
        
        # 6. 参数学习率分析
        if 'lr' in df.columns:
            ax7 = plt.subplot(3, 3, 7)
            lr_values = pd.to_numeric(df['lr'], errors='coerce')
            ax7.scatter(lr_values, df['AUROC'], alpha=0.6, s=50)
            ax7.set_xlabel('学习率')
            ax7.set_ylabel('AUROC')
            ax7.set_title('学习率 vs 性能')
            ax7.grid(True, alpha=0.3)
        
        # 7. Dropout 分析
        if 'dropout' in df.columns:
            ax8 = plt.subplot(3, 3, 8)
            dropout_values = pd.to_numeric(df['dropout'], errors='coerce')
            ax8.scatter(dropout_values, df['AUROC'], alpha=0.6, s=50, color='purple')
            ax8.set_xlabel('Dropout 率')
            ax8.set_ylabel('AUROC')
            ax8.set_title('Dropout vs 性能')
            ax8.grid(True, alpha=0.3)
        
        # 8. 批次大小分析
        if 'batch' in df.columns:
            ax9 = plt.subplot(3, 3, 9)
            batch_values = pd.to_numeric(df['batch'], errors='coerce')
            ax9.scatter(batch_values, df['AUROC'], alpha=0.6, s=50, color='brown')
            ax9.set_xlabel('批次大小')
            ax9.set_ylabel('AUROC')
            ax9.set_title('批次大小 vs 性能')
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_performance_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_parameter_correlation_analysis(self):
        """创建参数相关性分析"""
        if not self.iterations_data:
            return
            
        df = pd.DataFrame(self.iterations_data)
        
        # 选择数值型参数
        numeric_cols = []
        for col in df.columns:
            if col not in ['iteration', 'timestamp']:
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_cols.append(col)
                except:
                    pass
                    
        if len(numeric_cols) < 3:
            return
            
        # 转换为数值型
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 计算相关性矩阵
        correlation_matrix = df[numeric_cols].corr()
        
        # 创建相关性热力图
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('参数相关性分析热力图', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_correlation_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_optimization_landscape(self):
        """创建优化景观图"""
        if not self.iterations_data:
            return
            
        df = pd.DataFrame(self.iterations_data)
        
        # 选择两个主要参数进行3D可视化
        if 'lr' in df.columns and 'dropout' in df.columns:
            fig = plt.figure(figsize=(15, 10))
            
            # 3D 散点图
            ax1 = fig.add_subplot(121, projection='3d')
            lr_values = pd.to_numeric(df['lr'], errors='coerce')
            dropout_values = pd.to_numeric(df['dropout'], errors='coerce')
            
            scatter = ax1.scatter(lr_values, dropout_values, df['AUROC'], 
                                c=df['AUROC'], cmap='viridis', s=60, alpha=0.8)
            ax1.set_xlabel('学习率')
            ax1.set_ylabel('Dropout 率')
            ax1.set_zlabel('AUROC')
            ax1.set_title('优化景观 (学习率 vs Dropout)')
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax1, shrink=0.5)
            
            # 2D 等高线图
            ax2 = fig.add_subplot(122)
            
            # 创建网格进行插值
            from scipy.interpolate import griddata
            
            # 移除NaN值
            mask = ~(np.isnan(lr_values) | np.isnan(dropout_values) | np.isnan(df['AUROC']))
            lr_clean = lr_values[mask]
            dropout_clean = dropout_values[mask]
            auroc_clean = df['AUROC'][mask]
            
            if len(lr_clean) > 3:
                # 创建规则网格
                lr_grid = np.linspace(lr_clean.min(), lr_clean.max(), 50)
                dropout_grid = np.linspace(dropout_clean.min(), dropout_clean.max(), 50)
                LR_grid, DROPOUT_grid = np.meshgrid(lr_grid, dropout_grid)
                
                # 插值
                AUROC_grid = griddata((lr_clean, dropout_clean), auroc_clean, 
                                    (LR_grid, DROPOUT_grid), method='cubic', fill_value=0)
                
                # 绘制等高线
                contour = ax2.contourf(LR_grid, DROPOUT_grid, AUROC_grid, levels=20, cmap='viridis')
                ax2.scatter(lr_clean, dropout_clean, c=auroc_clean, cmap='viridis', 
                          s=60, edgecolors='white', linewidth=1)
                
                ax2.set_xlabel('学习率')
                ax2.set_ylabel('Dropout 率')
                ax2.set_title('优化景观等高线图')
                plt.colorbar(contour, ax=ax2)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'optimization_landscape.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
    def create_best_configurations_analysis(self):
        """分析最佳配置"""
        if not self.iterations_data:
            return
            
        df = pd.DataFrame(self.iterations_data)
        df = df.sort_values('AUROC', ascending=False)
        
        # 获取前10个最佳配置
        top_10 = df.head(10)
        
        # 创建最佳配置对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 最佳配置的性能指标对比
        ax1 = axes[0, 0]
        metrics = ['AUROC', 'AUPRC', 'F1', 'precision', 'recall']
        x_pos = np.arange(len(top_10))
        
        for i, metric in enumerate(metrics):
            if metric in top_10.columns:
                ax1.plot(x_pos, top_10[metric], 'o-', label=metric, linewidth=2, markersize=6)
        
        ax1.set_xlabel('配置排名 (按AUROC)')
        ax1.set_ylabel('指标值')
        ax1.set_title('Top 10 配置的性能指标对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 最佳配置的参数分布
        ax2 = axes[0, 1]
        if 'lr' in top_10.columns:
            lr_values = pd.to_numeric(top_10['lr'], errors='coerce')
            ax2.bar(x_pos, lr_values, alpha=0.7, color='blue')
            ax2.set_xlabel('配置排名')
            ax2.set_ylabel('学习率')
            ax2.set_title('Top 10 配置的学习率分布')
            ax2.grid(True, alpha=0.3)
        
        # 3. 评估时间对比
        ax3 = axes[1, 0]
        ax3.bar(x_pos, top_10['evaluation_time'], alpha=0.7, color='orange')
        ax3.set_xlabel('配置排名')
        ax3.set_ylabel('评估时间 (秒)')
        ax3.set_title('Top 10 配置的评估时间')
        ax3.grid(True, alpha=0.3)
        
        # 4. 损失值对比
        ax4 = axes[1, 1]
        ax4.bar(x_pos, top_10['loss'], alpha=0.7, color='red')
        ax4.set_xlabel('配置排名')
        ax4.set_ylabel('损失值')
        ax4.set_title('Top 10 配置的损失值')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'best_configurations_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存最佳配置详细信息
        self._save_best_configurations(top_10)
        
    def _save_best_configurations(self, top_configs):
        """保存最佳配置的详细信息"""
        with open(self.output_dir / 'best_configurations.txt', 'w', encoding='utf-8') as f:
            f.write("=== Top 10 最佳配置详细信息 ===\n\n")
            
            for i, (idx, config) in enumerate(top_configs.iterrows(), 1):
                f.write(f"排名 {i}:\n")
                f.write(f"  迭代次数: {config.get('iteration', 'N/A')}\n")
                f.write(f"  AUROC: {config.get('AUROC', 0):.6f}\n")
                f.write(f"  AUPRC: {config.get('AUPRC', 0):.6f}\n")
                f.write(f"  F1: {config.get('F1', 0):.6f}\n")
                f.write(f"  Precision: {config.get('precision', 0):.6f}\n")
                f.write(f"  Recall: {config.get('recall', 0):.6f}\n")
                f.write(f"  Loss: {config.get('loss', 0):.6f}\n")
                f.write(f"  评估时间: {config.get('evaluation_time', 0):.2f} 秒\n")
                
                # 添加关键参数
                key_params = ['lr', 'dropout', 'dimensions', 'batch', 'alpha', 'beta', 'gamma']
                f.write("  关键参数:\n")
                for param in key_params:
                    if param in config and pd.notna(config[param]):
                        f.write(f"    {param}: {config[param]}\n")
                f.write("\n")
                
    def generate_detailed_report(self):
        """生成详细分析报告"""
        if not self.iterations_data:
            return
            
        df = pd.DataFrame(self.iterations_data)
        
        report = []
        report.append("=== 详细优化分析报告 ===\n")
        
        # 基本统计信息
        report.append("## 基本统计信息")
        report.append(f"总迭代次数: {len(df)}")
        report.append(f"最佳AUROC: {df['AUROC'].max():.6f}")
        report.append(f"最差AUROC: {df['AUROC'].min():.6f}")
        report.append(f"平均AUROC: {df['AUROC'].mean():.6f}")
        report.append(f"AUROC标准差: {df['AUROC'].std():.6f}")
        report.append("")
        
        # 收敛分析
        report.append("## 收敛分析")
        best_iteration = df.loc[df['AUROC'].idxmax(), 'iteration']
        report.append(f"最佳结果出现在第 {best_iteration} 次迭代")
        
        # 计算收敛速度
        cummax = df['AUROC'].cummax()
        improvements = (cummax.diff() > 0).sum()
        report.append(f"总共有 {improvements} 次性能改进")
        report.append(f"平均每 {len(df)/improvements:.1f} 次迭代有一次改进")
        report.append("")
        
        # 参数分析
        report.append("## 关键参数分析")
        key_params = ['lr', 'dropout', 'dimensions', 'batch']
        for param in key_params:
            if param in df.columns:
                param_values = pd.to_numeric(df[param], errors='coerce')
                if not param_values.isna().all():
                    best_value = df.loc[df['AUROC'].idxmax(), param]
                    report.append(f"{param}:")
                    report.append(f"  最佳配置值: {best_value}")
                    report.append(f"  平均值: {param_values.mean():.6f}")
                    report.append(f"  标准差: {param_values.std():.6f}")
        report.append("")
        
        # 时间分析
        report.append("## 时间效率分析")
        total_time = df['evaluation_time'].sum()
        avg_time = df['evaluation_time'].mean()
        report.append(f"总优化时间: {total_time:.2f} 秒 ({total_time/3600:.2f} 小时)")
        report.append(f"平均每次评估: {avg_time:.2f} 秒")
        report.append(f"最快评估: {df['evaluation_time'].min():.2f} 秒")
        report.append(f"最慢评估: {df['evaluation_time'].max():.2f} 秒")
        report.append("")
        
        # 保存报告
        with open(self.output_dir / 'detailed_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        print('\n'.join(report))
        
    def run_complete_analysis(self):
        """运行完整的详细分析"""
        print("开始详细分析...")
        
        # 加载数据
        self.load_and_parse_data()
        
        # 生成各种分析图表
        print("1. 生成综合性能分析图...")
        self.create_comprehensive_performance_plot()
        
        print("2. 生成参数相关性分析...")
        self.create_parameter_correlation_analysis()
        
        print("3. 生成优化景观图...")
        self.create_optimization_landscape()
        
        print("4. 生成最佳配置分析...")
        self.create_best_configurations_analysis()
        
        print("5. 生成详细分析报告...")
        self.generate_detailed_report()
        
        print(f"\n详细分析完成！所有结果保存在: {self.output_dir}")

def main():
    """主函数"""
    log_dir = "logs/autodl_20260201_171815_20260201_171815"
    
    if not os.path.exists(log_dir):
        print(f"错误：日志目录 {log_dir} 不存在")
        return
        
    # 创建详细分析器并运行分析
    analyzer = DetailedAnalyzer(log_dir)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()