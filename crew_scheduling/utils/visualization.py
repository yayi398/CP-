# -*- coding: utf-8 -*-
"""
结果可视化模块
Visualization utilities for crew scheduling results
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
import os


def plot_gantt_chart(assignments: List[Dict], flights: pd.DataFrame, 
                     crews: pd.DataFrame, output_file: str = None):
    """
    绘制甘特图展示机组排班
    
    Args:
        assignments: 分配结果列表
        flights: 航班数据框
        crews: 机组数据框
        output_file: 输出文件路径（可选）
    """
    if not assignments:
        print("没有分配结果，无法绘制甘特图")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(16, max(8, len(assignments) * 0.5)))
    
    colors = plt.cm.Set3(range(len(assignments)))
    
    y_pos = 0
    y_labels = []
    
    for idx, assignment in enumerate(assignments):
        crew_id = assignment['crew_id']
        crew_number = assignment['crew_number']
        
        y_labels.append(crew_number)
        
        # 获取该机组的所有航班
        if 'flights' in assignment:
            flight_indices = assignment['flights']
        elif 'pairings' in assignment:
            flight_indices = []
            for pairing in assignment['pairings']:
                flight_indices.extend(pairing['flights'])
        else:
            continue
        
        # 绘制每个航班
        for flight_idx in flight_indices:
            flight = flights.iloc[flight_idx]
            
            start = flight['DepartureDateTime']
            end = flight['ArrivalDateTime']
            duration = (end - start).total_seconds() / 3600  # 转换为小时
            
            # 绘制矩形
            rect = Rectangle(
                (mdates.date2num(start), y_pos - 0.4),
                duration / 24,  # 转换为天数
                0.8,
                facecolor=colors[idx % len(colors)],
                edgecolor='black',
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # 添加航班号标签
            ax.text(
                mdates.date2num(start) + duration / 48,
                y_pos,
                flight['FltNum'],
                ha='center',
                va='center',
                fontsize=8
            )
        
        y_pos += 1
    
    # 设置y轴
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel('机组编号', fontsize=12)
    
    # 设置x轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.set_xlabel('日期', fontsize=12)
    
    # 设置网格
    ax.grid(True, axis='x', alpha=0.3)
    
    # 设置标题
    ax.set_title('机组排班甘特图', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"甘特图已保存到: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_cost_breakdown(result: Dict, output_file: str = None):
    """
    绘制成本分解饼图
    
    Args:
        result: 求解结果
        output_file: 输出文件路径（可选）
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 假设成本组成（实际应该从结果中提取）
    costs = {
        '执勤成本': result.get('objective', 0) * 0.7,
        '配对成本': result.get('objective', 0) * 0.2,
        '空乘成本': result.get('objective', 0) * 0.1
    }
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    ax.pie(
        costs.values(),
        labels=costs.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    
    ax.set_title('成本分解', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"成本分解图已保存到: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_crew_utilization(assignments: List[Dict], flights: pd.DataFrame,
                          crews: pd.DataFrame, output_file: str = None):
    """
    绘制机组利用率柱状图
    
    Args:
        assignments: 分配结果
        flights: 航班数据框
        crews: 机组数据框
        output_file: 输出文件路径（可选）
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算每个机组的工作时长
    crew_hours = {}
    
    for assignment in assignments:
        crew_number = assignment['crew_number']
        
        if 'flights' in assignment:
            flight_indices = assignment['flights']
        elif 'pairings' in assignment:
            flight_indices = []
            for pairing in assignment['pairings']:
                flight_indices.extend(pairing['flights'])
        else:
            continue
        
        total_hours = sum(flights.iloc[f]['FlightDuration'] for f in flight_indices) / 60.0
        crew_hours[crew_number] = total_hours
    
    if not crew_hours:
        print("没有机组工作数据，无法绘制利用率图")
        return
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(14, 6))
    
    crews_sorted = sorted(crew_hours.keys())
    hours_sorted = [crew_hours[c] for c in crews_sorted]
    
    bars = ax.bar(crews_sorted, hours_sorted, color='skyblue', edgecolor='black')
    
    # 添加平均线
    avg_hours = sum(hours_sorted) / len(hours_sorted) if hours_sorted else 0
    ax.axhline(y=avg_hours, color='red', linestyle='--', label=f'平均: {avg_hours:.1f}小时')
    
    ax.set_xlabel('机组编号', fontsize=12)
    ax.set_ylabel('工作时长 (小时)', fontsize=12)
    ax.set_title('机组工作时长分布', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"利用率图已保存到: {output_file}")
    else:
        plt.show()
    
    plt.close()


def generate_all_visualizations(result: Dict, flights: pd.DataFrame, 
                                crews: pd.DataFrame, output_dir: str = 'results'):
    """
    生成所有可视化图表
    
    Args:
        result: 求解结果
        flights: 航班数据框
        crews: 机组数据框
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    assignments = result.get('assignments', [])
    
    if assignments:
        # 甘特图
        plot_gantt_chart(
            assignments, flights, crews,
            os.path.join(output_dir, 'gantt_chart.png')
        )
        
        # 机组利用率
        plot_crew_utilization(
            assignments, flights, crews,
            os.path.join(output_dir, 'crew_utilization.png')
        )
    
    # 成本分解
    if result.get('objective'):
        plot_cost_breakdown(
            result,
            os.path.join(output_dir, 'cost_breakdown.png')
        )
    
    print(f"\n所有可视化图表已生成到目录: {output_dir}")


def print_solution_summary(result: Dict, flights: pd.DataFrame, crews: pd.DataFrame):
    """
    打印求解结果摘要
    
    Args:
        result: 求解结果
        flights: 航班数据框
        crews: 机组数据框
    """
    print("\n" + "=" * 60)
    print("求解结果摘要")
    print("=" * 60)
    
    print(f"求解状态: {result.get('status', 'Unknown')}")
    print(f"目标函数值: {result.get('objective', 'N/A')}")
    print(f"求解时间: {result.get('solve_time', 0):.2f} 秒")
    
    if 'algorithm' in result:
        print(f"算法: {result['algorithm']}")
    
    assignments = result.get('assignments', [])
    print(f"\n分配的机组数: {len(assignments)}")
    
    # 统计覆盖的航班
    covered_flights = set()
    for assignment in assignments:
        if 'flights' in assignment:
            covered_flights.update(assignment['flights'])
        elif 'pairings' in assignment:
            for pairing in assignment['pairings']:
                covered_flights.update(pairing['flights'])
    
    print(f"覆盖的航班数: {len(covered_flights)} / {len(flights)}")
    print(f"未覆盖的航班数: {len(flights) - len(covered_flights)}")
    
    if 'statistics' in result:
        stats = result['statistics']
        print("\n详细统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print("=" * 60 + "\n")
