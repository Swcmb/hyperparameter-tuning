#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析日志文件 logs/autodl_20260202_105812_20260202_105812
生成可视化图表、提取最佳参数并转换为命令行格式
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
plt.style.use('seaborn-v0_8')

class NewLogAnalyzer:
    def __init__(self, log_dir):
        """初始化分析器"""
        self.log_dir = Path(log_dir)
        self.structured_log_file = self.log_dir / "autodl_20260202_105812_structured.jsonl"
        self.main_log_file = self.log_dir / "autodl_20260202_105812_main.log"
        
        # 存储解析后的数据
        self.all_data = []
        self.iterations_data = []
        self.best_result = None
        
        # 输出目录
        self.output_dir = Path("analysis_20260202_105812")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_and_parse_data(self):
        """加载并解析所有数据"""
        print("正在加载和解析日志数据...")
        
        if not self.structured_log_file.exists():
            print(f"错误：找不到结构化日志文件 {self.structured_log_file}")
            return False
            
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
        return True
        
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
            
        # 找到最佳结果
        if self.iterations_data:
            self.best_result = max(self.iterations_data, 
                                 key=lambda x: x.get('AUROC', 0))
            
    def create_comprehensive_analysis(self):
        """创建综合分析图表"""
        if not self.iterations_data:
            return
            
        df = pd.DataFrame(self.iterations_data)
        df = df.sort_values('iteration')
        
        # 创建大图表
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 收敛曲线
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(df['iteration'], df['AUROC'], 'b-o', linewidth=2, markersize=4, label='AUROC')
        if 'AUPRC' in df.columns:
            ax1.plot(df['iteration'], df['AUPRC'], 'r-s', linewidth=2, markersize=4, label='AUPRC')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('性能指标')
        ax1.set_title('主要性能指标收敛曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 累积最佳值
        ax2 = plt.subplot(3, 3, 2)
        cumulative_best = df['AUROC'].cummax()
        ax2.plot(df['iteration'], df['AUROC'], 'lightblue', alpha=0.6, label='当前值')
        ax2.plot(df['iteration'], cumulative_best, 'darkblue', linewidth=3, label='累积最佳')
        ax2.fill_between(df['iteration'], df['AUROC'], cumulative_best, alpha=0.3)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('AUROC')
        ax2.set_title('AUROC累积最佳值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 评估时间分析
        ax3 = plt.subplot(3, 3, 3)
        ax3.bar(df['iteration'], df['evaluation_time'], alpha=0.7, color='orange')
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('评估时间 (秒)')
        ax3.set_title('每次迭代评估时间')
        ax3.grid(True, alpha=0.3)
        
        # 4. 多指标对比
        ax4 = plt.subplot(3, 3, 4)
        metrics = ['F1', 'precision', 'recall']
        for metric in metrics:
            if metric in df.columns:
                ax4.plot(df['iteration'], df[metric], 'o-', label=metric, linewidth=2, markersize=4)
        ax4.set_xlabel('迭代次数')
        ax4.set_ylabel('指标值')
        ax4.set_title('分类性能指标')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 损失函数
        ax5 = plt.subplot(3, 3, 5)
        if 'loss' in df.columns:
            ax5.plot(df['iteration'], df['loss'], 'r-o', linewidth=2, markersize=4)
            ax5.set_xlabel('迭代次数')
            ax5.set_ylabel('损失值')
            ax5.set_title('损失函数变化')
            ax5.grid(True, alpha=0.3)
        
        # 6. 性能改进分析
        ax6 = plt.subplot(3, 3, 6)
        improvements = df['AUROC'].diff().fillna(0)
        colors = ['green' if x > 0 else 'red' for x in improvements]
        ax6.bar(df['iteration'], improvements, color=colors, alpha=0.7)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.set_xlabel('迭代次数')
        ax6.set_ylabel('AUROC改进量')
        ax6.set_title('每次迭代的性能改进')
        ax6.grid(True, alpha=0.3)
        
        # 7-9. 参数分析
        param_plots = [
            ('lr', '学习率'),
            ('dropout', 'Dropout率'),
            ('batch', '批次大小')
        ]
        
        for i, (param, title) in enumerate(param_plots, 7):
            if param in df.columns:
                ax = plt.subplot(3, 3, i)
                param_values = pd.to_numeric(df[param], errors='coerce')
                ax.scatter(param_values, df['AUROC'], alpha=0.6, s=50)
                ax.set_xlabel(title)
                ax.set_ylabel('AUROC')
                ax.set_title(f'{title} vs 性能')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_parameter_analysis(self):
        """创建参数分析图表"""
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
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('参数相关性分析热力图', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_correlation.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def extract_best_parameters(self):
        """提取最佳参数"""
        if not self.best_result:
            print("没有找到最佳结果")
            return None
            
        print(f"\n=== 最佳结果分析 ===")
        print(f"最佳迭代: 第{self.best_result['iteration']}次")
        print(f"最佳AUROC: {self.best_result.get('AUROC', 0):.6f}")
        print(f"AUPRC: {self.best_result.get('AUPRC', 0):.6f}")
        print(f"F1 Score: {self.best_result.get('F1', 0):.6f}")
        print(f"Precision: {self.best_result.get('precision', 0):.6f}")
        print(f"Recall: {self.best_result.get('recall', 0):.6f}")
        print(f"Loss: {self.best_result.get('loss', 0):.6f}")
        print(f"评估时间: {self.best_result.get('evaluation_time', 0):.2f}秒")
        
        # 提取参数
        best_params = {}
        param_keys = [
            'dimensions', 'hidden1', 'hidden2', 'decoder1',
            'lr', 'dropout', 'weight_decay', 'batch',
            'alpha', 'beta', 'gamma',
            'moco_momentum', 'moco_t', 'moco_tau1', 'moco_tau2', 'moco_K', 'moco_type',
            'gat_heads', 'gt_heads', 'fusion_heads', 'fusion_strategy',
            'feature_type', 'enable_view_0'
        ]
        
        for key in param_keys:
            if key in self.best_result:
                best_params[key] = self.best_result[key]
        
        # 保存最佳参数
        with open(self.output_dir / 'best_parameters.json', 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
            
        return best_params
        
    def generate_command_line(self, best_params):
        """生成命令行格式"""
        if not best_params:
            return
            
        print(f"\n=== 最佳参数命令行格式 ===")
        
        # 参数类型修正
        def fix_param_value(param_name, param_value):
            int_params = [
                'dimensions', 'hidden1', 'hidden2', 'decoder1', 
                'batch', 'gat_heads', 'gt_heads', 'fusion_heads', 'moco_K'
            ]
            
            float_params = [
                'lr', 'dropout', 'weight_decay', 'alpha', 'beta', 'gamma',
                'moco_momentum', 'moco_t', 'moco_tau1', 'moco_tau2'
            ]
            
            bool_params = ['enable_view_0']
            str_params = ['fusion_strategy', 'feature_type', 'moco_type']
            
            if param_name in int_params:
                return str(int(float(param_value)))
            elif param_name in float_params:
                return str(float(param_value))
            elif param_name in bool_params:
                return 'true' if str(param_value).lower() in ['true', '1', 'yes'] else 'false'
            else:
                return str(param_value)
        
        # 参数映射
        param_mapping = {
            'dimensions': '--dimensions',
            'hidden1': '--hidden1', 
            'hidden2': '--hidden2',
            'decoder1': '--decoder1',
            'lr': '--lr',
            'dropout': '--dropout',
            'weight_decay': '--weight_decay',
            'batch': '--batch',
            'moco_momentum': '--moco_momentum',
            'moco_t': '--moco_t',
            'moco_tau1': '--moco_tau1',
            'moco_tau2': '--moco_tau2',
            'moco_K': '--moco_K',
            'moco_type': '--moco_type',
            'gat_heads': '--gat_heads',
            'gt_heads': '--gt_heads', 
            'fusion_heads': '--fusion_heads',
            'fusion_strategy': '--fusion_strategy',
            'alpha': '--alpha',
            'beta': '--beta',
            'gamma': '--gamma',
            'feature_type': '--feature_type',
            'enable_view_0': '--enable_view_0'
        }
        
        # 生成命令行
        base_command = "python main.py"
        all_params = []
        
        for param_name, param_value in best_params.items():
            if param_name in param_mapping:
                fixed_value = fix_param_value(param_name, param_value)
                all_params.append(f"{param_mapping[param_name]} {fixed_value}")
        
        # 添加基础参数
        extra_params = [
            "--task_type LDA",
            "--run_name BEST_20260202", 
            "--seed 42"
        ]
        
        full_command = f"{base_command} " + " ".join(all_params) + " " + " ".join(extra_params)
        
        print("\n## 完整命令行:")
        print("```bash")
        print(full_command)
        print("```")
        
        # 分类显示
        print("\n## 分类参数命令:")
        print("```bash")
        print(f"{base_command} \\")
        print("  --task_type LDA \\")
        print("  --run_name BEST_20260202 \\")
        print("  --seed 42 \\")
        
        # 按类别组织参数
        categories = {
            '网络架构': ['dimensions', 'hidden1', 'hidden2', 'decoder1'],
            '优化参数': ['lr', 'dropout', 'weight_decay', 'batch'],
            'MoCo参数': ['moco_momentum', 'moco_t', 'moco_tau1', 'moco_tau2', 'moco_K', 'moco_type'],
            '注意力机制': ['gat_heads', 'gt_heads', 'fusion_heads', 'fusion_strategy'],
            '其他参数': ['alpha', 'beta', 'gamma', 'feature_type', 'enable_view_0']
        }
        
        for category, params in categories.items():
            if any(p in best_params for p in params):
                print(f"  # {category}")
                for param in params:
                    if param in best_params:
                        fixed_value = fix_param_value(param, best_params[param])
                        print(f"  {param_mapping[param]} {fixed_value} \\")
        
        print("```")
        
        # 保存命令行
        with open(self.output_dir / 'best_command.sh', 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n")
            f.write("# 最佳参数命令行 - 20260202_105812\n")
            f.write(f"# AUROC: {self.best_result.get('AUROC', 0):.6f}\n\n")
            f.write(full_command + "\n")
            
    def generate_summary_report(self):
        """生成总结报告"""
        if not self.iterations_data:
            return
            
        df = pd.DataFrame(self.iterations_data)
        
        report = []
        report.append("=== 优化过程分析报告 (20260202_105812) ===\n")
        
        # 基本统计
        report.append("## 基本统计信息")
        report.append(f"总迭代次数: {len(df)}")
        report.append(f"最佳AUROC: {df['AUROC'].max():.6f}")
        report.append(f"最差AUROC: {df['AUROC'].min():.6f}")
        report.append(f"平均AUROC: {df['AUROC'].mean():.6f}")
        report.append(f"AUROC标准差: {df['AUROC'].std():.6f}")
        report.append("")
        
        # 最佳结果
        if self.best_result:
            report.append("## 最佳结果")
            report.append(f"最佳迭代: 第{self.best_result['iteration']}次")
            report.append(f"AUROC: {self.best_result.get('AUROC', 0):.6f}")
            report.append(f"AUPRC: {self.best_result.get('AUPRC', 0):.6f}")
            report.append(f"F1 Score: {self.best_result.get('F1', 0):.6f}")
            report.append(f"评估时间: {self.best_result.get('evaluation_time', 0):.2f}秒")
            report.append("")
        
        # 时间分析
        report.append("## 时间效率")
        total_time = df['evaluation_time'].sum()
        avg_time = df['evaluation_time'].mean()
        report.append(f"总优化时间: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
        report.append(f"平均评估时间: {avg_time:.2f}秒")
        report.append("")
        
        # 保存报告
        with open(self.output_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        print('\n'.join(report))
        
    def run_complete_analysis(self):
        """运行完整分析"""
        print("开始分析日志 autodl_20260202_105812...")
        
        # 加载数据
        if not self.load_and_parse_data():
            return
        
        # 生成分析图表
        print("1. 生成综合分析图表...")
        self.create_comprehensive_analysis()
        
        print("2. 生成参数相关性分析...")
        self.create_parameter_analysis()
        
        print("3. 提取最佳参数...")
        best_params = self.extract_best_parameters()
        
        print("4. 生成命令行格式...")
        self.generate_command_line(best_params)
        
        print("5. 生成分析报告...")
        self.generate_summary_report()
        
        print(f"\n分析完成！所有结果保存在: {self.output_dir}")

def main():
    """主函数"""
    log_dir = "logs/autodl_20260202_105812_20260202_105812"
    
    if not os.path.exists(log_dir):
        print(f"错误：日志目录 {log_dir} 不存在")
        return
        
    # 创建分析器并运行分析
    analyzer = NewLogAnalyzer(log_dir)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()