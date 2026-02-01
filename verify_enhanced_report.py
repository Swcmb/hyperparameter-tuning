#!/usr/bin/env python3
"""
验证增强版报告的完整性
"""

import os
from pathlib import Path

def verify_enhanced_report():
    """验证增强版报告的完整性"""
    
    print("🔍 验证增强版报告...")
    
    # 检查报告目录
    report_dir = Path("enhanced_reports")
    if not report_dir.exists():
        print("❌ 报告目录不存在")
        return False
    
    # 检查主要报告文件
    html_file = report_dir / "optimization_report_enhanced.html"
    json_file = report_dir / "optimization_report_enhanced.json"
    
    if not html_file.exists():
        print("❌ HTML报告文件不存在")
        return False
    
    if not json_file.exists():
        print("❌ JSON报告文件不存在")
        return False
    
    print("✅ 主要报告文件存在")
    
    # 检查图表目录
    charts_dir = report_dir / "charts"
    if not charts_dir.exists():
        print("❌ 图表目录不存在")
        return False
    
    # 检查所有预期的图表文件
    expected_charts = [
        "convergence_curve.png",
        "parameter_importance.png", 
        "objective_distribution.png",
        "parameter_correlation_heatmap.png",
        "performance_heatmap.png",
        "parameter_distributions.png",
        "parameter_evolution.png",
        "optimization_landscape_3d.png"
    ]
    
    missing_charts = []
    existing_charts = []
    
    for chart_name in expected_charts:
        chart_path = charts_dir / chart_name
        if chart_path.exists():
            existing_charts.append(chart_name)
            print(f"✅ {chart_name}")
        else:
            missing_charts.append(chart_name)
            print(f"❌ {chart_name} 缺失")
    
    # 检查HTML文件中的图表引用
    print("\n🔍 检查HTML文件中的图表引用...")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    chart_references_found = 0
    for chart_name in existing_charts:
        chart_ref = f'charts/{chart_name}'
        if chart_ref in html_content:
            chart_references_found += 1
            print(f"✅ HTML中引用了 {chart_name}")
        else:
            print(f"❌ HTML中未引用 {chart_name}")
    
    # 检查是否包含热力图
    heatmap_charts = [chart for chart in existing_charts if 'heatmap' in chart]
    print(f"\n🔥 热力图文件: {len(heatmap_charts)} 个")
    for heatmap in heatmap_charts:
        print(f"   ✅ {heatmap}")
    
    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"   📁 报告目录: {report_dir}")
    print(f"   📄 HTML报告: {html_file}")
    print(f"   📋 JSON报告: {json_file}")
    print(f"   📊 图表目录: {charts_dir}")
    print(f"   🖼️  生成的图表: {len(existing_charts)}/{len(expected_charts)}")
    print(f"   🔗 HTML引用: {chart_references_found}/{len(existing_charts)}")
    print(f"   🔥 热力图: {len(heatmap_charts)} 个")
    
    # 文件大小信息
    print(f"\n📏 文件大小:")
    print(f"   HTML: {html_file.stat().st_size / 1024:.1f} KB")
    print(f"   JSON: {json_file.stat().st_size / 1024:.1f} KB")
    
    total_chart_size = 0
    for chart_name in existing_charts:
        chart_path = charts_dir / chart_name
        size = chart_path.stat().st_size
        total_chart_size += size
        print(f"   {chart_name}: {size / 1024:.1f} KB")
    
    print(f"   图表总大小: {total_chart_size / 1024:.1f} KB")
    
    # 总结
    success = (len(missing_charts) == 0 and 
              chart_references_found == len(existing_charts) and
              len(heatmap_charts) >= 2)  # 至少要有2个热力图
    
    if success:
        print(f"\n🎉 增强版报告验证成功！")
        print(f"   ✅ 所有图表文件已生成")
        print(f"   ✅ HTML正确引用了图表文件")
        print(f"   ✅ 包含热力图")
        print(f"   ✅ 图表保存为单独文件（非内嵌base64）")
    else:
        print(f"\n⚠️  增强版报告存在问题:")
        if missing_charts:
            print(f"   ❌ 缺失图表: {missing_charts}")
        if chart_references_found != len(existing_charts):
            print(f"   ❌ HTML引用不完整")
        if len(heatmap_charts) < 2:
            print(f"   ❌ 热力图数量不足")
    
    return success

if __name__ == "__main__":
    verify_enhanced_report()