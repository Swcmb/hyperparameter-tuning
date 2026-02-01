#!/usr/bin/env python3
"""
生成增强版报告的脚本
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, '.')

try:
    from enhanced_report_generator import EnhancedReportGenerator, ReportConfig, create_enhanced_report_from_checkpoint
    from autodl_core import OptimizationHistory, ParameterSpace
    from state_manager import StateManager
    
    def main():
        # 查找最新的检查点文件
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            print("错误：检查点目录不存在")
            return
        
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
        if not checkpoint_files:
            print("错误：未找到检查点文件")
            return
        
        # 选择最新的检查点文件
        latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoint_files)[-1])
        print(f"使用检查点文件: {latest_checkpoint}")
        
        # 创建输出目录
        output_dir = "enhanced_reports"
        os.makedirs(output_dir, exist_ok=True)
        
        # 配置报告
        config = ReportConfig(
            title="LDA任务贝叶斯超参数优化报告（增强版）",
            author="AutoDL优化系统",
            include_charts=True,
            include_parameter_details=True,
            include_convergence_analysis=True,
            include_sensitivity_analysis=True,
            include_best_parameters_analysis=True
        )
        
        try:
            # 加载检查点数据
            state_manager = StateManager()
            state_data = state_manager.load_state(latest_checkpoint)
            
            # 检查优化历史数据（可能是 'history' 或 'optimization_history'）
            history_data = None
            if 'optimization_history' in state_data:
                history_data = state_data['optimization_history']
            elif 'history' in state_data:
                history_data = state_data['history']
            else:
                print("错误：检查点文件中没有优化历史数据")
                print(f"可用的键: {list(state_data.keys())}")
                return
            
            # 创建优化历史对象
            if isinstance(history_data, dict):
                history = OptimizationHistory.from_dict(history_data)
            else:
                # 如果是对象，直接使用
                history = history_data
            print(f"加载了 {len(history.results)} 个优化结果")
            
            # 创建参数空间对象（如果存在）
            parameter_space = None
            if 'parameter_space' in state_data:
                param_space_data = state_data['parameter_space']
                if isinstance(param_space_data, dict):
                    parameter_space = ParameterSpace.from_dict(param_space_data)
                else:
                    # 如果是对象，直接使用
                    parameter_space = param_space_data
                print(f"加载了参数空间，包含 {parameter_space.get_parameter_count()} 个参数")
            
            # 创建增强报告生成器
            generator = EnhancedReportGenerator(history, parameter_space, config=config)
            
            # 生成增强报告
            print("开始生成增强版报告...")
            generator.generate_enhanced_report(output_dir, "optimization_report_enhanced")
            
            print(f"\n✅ 增强版报告生成完成！")
            print(f"📁 报告目录: {output_dir}")
            print(f"📊 图表目录: {os.path.join(output_dir, 'charts')}")
            print(f"📄 HTML报告: {os.path.join(output_dir, 'optimization_report_enhanced.html')}")
            print(f"📋 JSON报告: {os.path.join(output_dir, 'optimization_report_enhanced.json')}")
            
        except Exception as e:
            print(f"生成报告时出错: {e}")
            import traceback
            traceback.print_exc()
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有必要的模块都已正确安装")