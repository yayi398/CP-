#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整示例：运行优化并导出结果
Complete example: Run optimization and export results
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crew_scheduling.data_loader import DataLoader
from crew_scheduling.solvers.pso_solver import PSOSolver
from crew_scheduling import config


def export_results_example():
    """演示如何导出结果到CSV"""
    
    print("=" * 80)
    print("机组排班优化 - 结果导出示例")
    print("=" * 80)
    
    # 加载数据
    print("\n1. 加载数据...")
    loader = DataLoader(config.CREW_DATA_FILE, config.FLIGHT_DATA_FILE)
    crews, flights = loader.load_data()
    print(f"   加载了 {len(crews)} 名机组和 {len(flights)} 个航班")
    
    # 配置参数（使用较小的参数以加快演示）
    demo_config = {
        'PSO_SWARM_SIZE': 30,
        'PSO_ITERATIONS': 20,
        'PSO_W': 0.7,
        'PSO_C1': 1.5,
        'PSO_C2': 1.5,
    }
    
    # 使用前60个航班进行演示
    flights_subset = flights[:60]
    
    # 求解
    print(f"\n2. 使用PSO求解（使用前{len(flights_subset)}个航班）...")
    solver = PSOSolver(flights_subset, crews, demo_config)
    result = solver.solve()
    
    print(f"   求解完成: 目标值={result['objective']:.2f}, "
          f"耗时={result['solve_time']:.2f}秒")
    
    # 导出机组排班表
    print("\n3. 导出结果到CSV...")
    
    import pandas as pd
    
    # 准备排班表数据
    roster_rows = []
    for assignment in result['assignments']:
        crew_id = assignment['crew_id']
        crew_number = assignment['crew_number']
        
        for flight_idx in assignment['flights']:
            flight = flights_subset.iloc[flight_idx]
            
            roster_rows.append({
                '机组编号': crew_number,
                '航班号': flight['FltNum'],
                '出发日期': flight['DptrDate'],
                '出发时间': flight['DptrTime'],
                '出发机场': flight['DptrStn'],
                '到达日期': flight['ArrvDate'],
                '到达时间': flight['ArrvTime'],
                '到达机场': flight['ArrvStn'],
                '飞行时长(分钟)': flight['FlightDuration'],
            })
    
    # 创建输出目录
    os.makedirs('demo_output', exist_ok=True)
    
    # 保存排班表
    roster_df = pd.DataFrame(roster_rows)
    roster_df = roster_df.sort_values(['机组编号', '出发日期', '出发时间'])
    roster_file = 'demo_output/crew_rosters_demo.csv'
    roster_df.to_csv(roster_file, index=False, encoding='utf-8-sig')
    
    print(f"   ✓ 排班表已保存到: {roster_file}")
    print(f"   ✓ 共 {len(roster_df)} 条分配记录")
    
    # 统计信息
    print("\n4. 统计信息:")
    
    # 按机组统计
    crew_stats = roster_df.groupby('机组编号').agg({
        '航班号': 'count',
        '飞行时长(分钟)': 'sum'
    }).rename(columns={'航班号': '航班数', '飞行时长(分钟)': '总飞行时长(分钟)'})
    
    crew_stats['总飞行时长(小时)'] = crew_stats['总飞行时长(分钟)'] / 60.0
    
    stats_file = 'demo_output/crew_statistics_demo.csv'
    crew_stats.to_csv(stats_file, encoding='utf-8-sig')
    
    print(f"   ✓ 机组统计已保存到: {stats_file}")
    
    # 显示前几条
    print("\n   机组工作统计 (前5名):")
    print(crew_stats.head().to_string())
    
    # 总体统计
    print(f"\n5. 总体统计:")
    print(f"   - 总成本: {result['objective']:.2f} 元")
    print(f"   - 参与机组数: {len(result['assignments'])}")
    print(f"   - 总航班数: {len(roster_df)}")
    print(f"   - 平均每机组航班数: {len(roster_df)/len(result['assignments']):.1f}")
    print(f"   - 平均飞行时长: {crew_stats['总飞行时长(小时)'].mean():.1f} 小时")
    print(f"   - 最大飞行时长: {crew_stats['总飞行时长(小时)'].max():.1f} 小时")
    print(f"   - 最小飞行时长: {crew_stats['总飞行时长(小时)'].min():.1f} 小时")
    
    print("\n" + "=" * 80)
    print("✓ 演示完成! 结果文件保存在 demo_output/ 目录")
    print("=" * 80)


if __name__ == '__main__':
    export_results_example()
