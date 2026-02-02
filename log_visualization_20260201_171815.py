#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于日志文件 logs/autodl_20260201_171815_20260201_171815 的可视化分析脚本
生成各种图表来展示优化过程的详细信息
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LogVisualizer:
    def __init__(self, log_dir):
        """
        初始化日志可视化器
        
        Args:
            log_dir: 日志文件夹路径
        """
        self.log_dir = Path(log_dir)
        # 修正文件名格式
        base_name = "autodl_20260201_171815"
        self.structured_log_file = self.log_dir / f"{base_name}_structured.jsonl"
        self.main_log_file = self.log_dir / f"{base_name}_main.log"
        
        # 存储解析后的数据
        self.optimization_data = []
        self.parameter_data = []
        self.evaluation_data = []
        self.timing_data = []
        
        # 输出目录
        self.output_dir = Path(f"visualization_output_{self.log_dir.name}")
        self.output_dir.mkdir(exist_ok=True)
        
    def parse_structured_logs(self):
        """解析结构化日志文件"""
        print("正在解析结构化日志文件...")
        
        if not self.structured_log_file.exists():
            print(f"错误：找不到结构化日志文件 {self.structured_log_file}")
            return
            
        with open(self.structured_log_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    self._process_log_entry(data)
                except json.JSONDecodeError as e:
                    print(f"第 {line_num} 行JSON解析错误: {e}")
                except Exception as e:
                    print(f"第 {line_num} 行处理错误: {e}")
                    
        print(f"解析完成：")
        print(f"  - 优化数据条目: {len(self.optimization_data)}")
        print(f"  - 参数数据条目: {len(self.parameter_data)}")
        print(f"  - 评估数据条目: {len(self.evaluation_data)}")
        print(f"  - 时间数据条目: {len(self.timing_data)}")
        
    def _process_log_entry(self, data):
        """处理单个日志条目"""
        tag = data.get('tag', '')
        timestamp = data.get('timestamp', '')
        
        # 处理不同类型的日志条目
        if tag == 'NEW_RESULT' and 'structured_data' in data:
            self._process_evaluation_result(data, timestamp)
        elif tag == 'SUGGESTED_PARAMS' and 'structured_data' in data:
            self._process_parameter_suggestion(data, timestamp)
        elif tag == 'ITERATION':
            self._process_iteration(data, timestamp)
        elif tag in ['SUGGESTION_INFO', 'UPDATE_SUMMARY']:
            self._process_timing_info(data, timestamp)
            
    def _process_evaluation_result(self, data, timestamp):
        """处理评估结果"""
        structured = data['structured_data']
        
        eval_data = {
            'timestamp': timestamp,
            'iteration': structured.get('iteration', 0),
            'objective_value': structured.get('objective_value', 0),
            'evaluation_time': structured.get('evaluation_time', 0),
            'parameter_count': structured.get('parameter_count', 0)
        }
        
        # 提取主要指标
        main_metrics = structured.get('main_metrics', {})
        for metric, value in main_metrics.items():
            eval_data[f'metric_{metric}'] = value
            
        self.evaluation_data.append(eval_data)
        
    def _process_parameter_suggestion(self, data, timestamp):
        """处理参数建议"""
        structured = data['structured_data']
        
        param_data = {
            'timestamp': timestamp,
            **structured
        }
        
        self.parameter_data.append(param_data)
        
    def _process_iteration(self, data, timestamp):
        """处理迭代信息"""
        message = data.get('message', '')
        
        opt_data = {
            'timestamp': timestamp,
            'iteration': message if message.isdigit() else 0
        }
        
        self.optimization_data.append(opt_data)
        
    def _process_timing_info(self, data, timestamp):
        """处理时间信息"""
        if 'structured_data' in data:
            timing_data = {
                'timestamp': timestamp,
                'tag': data.get('tag', ''),
                **data['structured_data']
            }
            self.timing_data.append(timing_data)
            
    def create_convergence_plot(self):
        """创建收敛曲线图"""
        if not self.evaluation_data:
            print("没有评估数据，跳过收敛曲线图")
            return
            
        df = pd.DataFrame(self.evaluation_data)
        df = df.sort_values('iteration')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 目标值收敛曲线
        ax1.plot(df['iteration'], df['objective_value'], 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('目标值 (AUROC)')
        ax1.set_title('优化收敛曲线 - 目标值变化')
        ax1.grid(True, alpha=0.3)
        
        # 计算累积最佳值
        cumulative_best = df['objective_value'].cummax()
        ax1.plot(df['iteration'], cumulative_best, 'r--', linewidth=2, label='累积最佳值')
        ax1.legend()
        
        # 评估时间变化
        ax2.plot(df['iteration'], df['evaluation_time'], 'g-s', linewidth=2, markersize=6)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('评估时间 (秒)')
        ax2.set_title('每次迭代的评估时间')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_metrics_comparison(self):
        """创建多指标对比图"""
        if not self.evaluation_data:
            print("没有评估数据，跳过指标对比图")
            return
            
        df = pd.DataFrame(self.evaluation_data)
        df = df.sort_values('iteration')
        
        # 提取所有指标列
        metric_cols = [col for col in df.columns if col.startswith('metric_')]
        
        if not metric_cols:
            print("没有找到指标数据")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, metric_col in enumerate(metric_cols[:4]):  # 只显示前4个指标
            metric_name = metric_col.replace('metric_', '')
            ax = axes[i]
            
            ax.plot(df['iteration'], df[metric_col], 
                   color=colors[i % len(colors)], 
                   marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} 变化趋势')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_parameter_distribution(self):
        """创建参数分布图"""
        if not self.parameter_data:
            print("没有参数数据，跳过参数分布图")
            return
            
        df = pd.DataFrame(self.parameter_data)
        
        # 选择数值型参数
        numeric_params = []
        for col in df.columns:
            if col != 'timestamp':
                try:
                    pd.to_numeric(df[col])
                    numeric_params.append(col)
                except:
                    pass
                    
        if len(numeric_params) < 4:
            print("数值型参数不足，跳过参数分布图")
            return
            
        # 选择前8个数值参数
        selected_params = numeric_params[:8]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, param in enumerate(selected_params):
            ax = axes[i]
            
            # 转换为数值型
            values = pd.to_numeric(df[param], errors='coerce').dropna()
            
            if len(values) > 0:
                ax.hist(values, bins=min(10, len(values)), alpha=0.7, edgecolor='black')
                ax.set_xlabel(param)
                ax.set_ylabel('频次')
                ax.set_title(f'{param} 分布')
                ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_parameter_evolution(self):
        """创建参数演化图"""
        if not self.parameter_data:
            print("没有参数数据，跳过参数演化图")
            return
            
        df = pd.DataFrame(self.parameter_data)
        
        # 添加迭代编号
        df['iteration'] = range(1, len(df) + 1)
        
        # 选择几个关键参数
        key_params = ['lr', 'dropout', 'dimensions', 'batch']
        available_params = [p for p in key_params if p in df.columns]
        
        if not available_params:
            print("没有找到关键参数，跳过参数演化图")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, param in enumerate(available_params[:4]):
            ax = axes[i]
            
            # 转换为数值型
            values = pd.to_numeric(df[param], errors='coerce')
            
            ax.plot(df['iteration'], values, 
                   color=colors[i], marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel(param)
            ax.set_title(f'{param} 参数演化')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_timing_analysis(self):
        """创建时间分析图"""
        if not self.timing_data:
            print("没有时间数据，跳过时间分析图")
            return
            
        df = pd.DataFrame(self.timing_data)
        
        # 分析建议时间
        suggestion_data = df[df['tag'] == 'SUGGESTION_INFO']
        if not suggestion_data.empty and 'suggestion_time' in suggestion_data.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 建议时间分布
            suggestion_times = pd.to_numeric(suggestion_data['suggestion_time'], errors='coerce').dropna()
            ax1.hist(suggestion_times, bins=10, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('建议时间 (秒)')
            ax1.set_ylabel('频次')
            ax1.set_title('参数建议时间分布')
            ax1.grid(True, alpha=0.3)
            
            # 建议时间趋势
            ax2.plot(range(len(suggestion_times)), suggestion_times, 'b-o', linewidth=2, markersize=6)
            ax2.set_xlabel('建议次数')
            ax2.set_ylabel('建议时间 (秒)')
            ax2.set_title('参数建议时间趋势')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'timing_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    def create_performance_heatmap(self):
        """创建性能热力图"""
        if not self.evaluation_data or not self.parameter_data:
            print("数据不足，跳过性能热力图")
            return
            
        eval_df = pd.DataFrame(self.evaluation_data)
        param_df = pd.DataFrame(self.parameter_data)
        
        # 合并数据
        if len(eval_df) != len(param_df):
            min_len = min(len(eval_df), len(param_df))
            eval_df = eval_df.head(min_len)
            param_df = param_df.head(min_len)
            
        # 选择数值型参数
        numeric_params = []
        for col in param_df.columns:
            if col != 'timestamp':
                try:
                    pd.to_numeric(param_df[col])
                    numeric_params.append(col)
                except:
                    pass
                    
        if len(numeric_params) < 2:
            print("数值型参数不足，跳过性能热力图")
            return
            
        # 创建相关性矩阵
        combined_df = param_df[numeric_params[:6]].copy()  # 选择前6个参数
        for col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            
        combined_df['objective_value'] = eval_df['objective_value']
        
        # 计算相关性
        correlation_matrix = combined_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('参数与目标值相关性热力图')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_summary_report(self):
        """创建总结报告"""
        print("\n=== 优化过程总结报告 ===")
        
        if self.evaluation_data:
            eval_df = pd.DataFrame(self.evaluation_data)
            print(f"总迭代次数: {len(eval_df)}")
            print(f"最佳目标值: {eval_df['objective_value'].max():.6f}")
            print(f"最差目标值: {eval_df['objective_value'].min():.6f}")
            print(f"平均目标值: {eval_df['objective_value'].mean():.6f}")
            print(f"目标值标准差: {eval_df['objective_value'].std():.6f}")
            
            if 'evaluation_time' in eval_df.columns:
                print(f"总评估时间: {eval_df['evaluation_time'].sum():.2f} 秒")
                print(f"平均评估时间: {eval_df['evaluation_time'].mean():.2f} 秒")
                
        if self.parameter_data:
            print(f"参数建议次数: {len(self.parameter_data)}")
            
        # 保存报告到文件
        with open(self.output_dir / 'summary_report.txt', 'w', encoding='utf-8') as f:
            f.write("=== 优化过程总结报告 ===\n\n")
            if self.evaluation_data:
                eval_df = pd.DataFrame(self.evaluation_data)
                f.write(f"总迭代次数: {len(eval_df)}\n")
                f.write(f"最佳目标值: {eval_df['objective_value'].max():.6f}\n")
                f.write(f"最差目标值: {eval_df['objective_value'].min():.6f}\n")
                f.write(f"平均目标值: {eval_df['objective_value'].mean():.6f}\n")
                f.write(f"目标值标准差: {eval_df['objective_value'].std():.6f}\n")
                
                if 'evaluation_time' in eval_df.columns:
                    f.write(f"总评估时间: {eval_df['evaluation_time'].sum():.2f} 秒\n")
                    f.write(f"平均评估时间: {eval_df['evaluation_time'].mean():.2f} 秒\n")
                    
            if self.parameter_data:
                f.write(f"参数建议次数: {len(self.parameter_data)}\n")
                
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print(f"开始生成可视化图表，输出目录: {self.output_dir}")
        
        # 解析日志
        self.parse_structured_logs()
        
        # 生成各种图表
        print("\n1. 生成收敛曲线图...")
        self.create_convergence_plot()
        
        print("2. 生成指标对比图...")
        self.create_metrics_comparison()
        
        print("3. 生成参数分布图...")
        self.create_parameter_distribution()
        
        print("4. 生成参数演化图...")
        self.create_parameter_evolution()
        
        print("5. 生成时间分析图...")
        self.create_timing_analysis()
        
        print("6. 生成性能热力图...")
        self.create_performance_heatmap()
        
        print("7. 生成总结报告...")
        self.create_summary_report()
        
        print(f"\n所有可视化图表已生成完成！输出目录: {self.output_dir}")

def main():
    """主函数"""
    log_dir = "logs/autodl_20260201_171815_20260201_171815"
    
    if not os.path.exists(log_dir):
        print(f"错误：日志目录 {log_dir} 不存在")
        return
        
    # 创建可视化器并生成图表
    visualizer = LogVisualizer(log_dir)
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()