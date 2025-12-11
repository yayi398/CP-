# -*- coding: utf-8 -*-
"""
主程序入口
Main entry point for crew scheduling optimization
"""

import sys
import os
import pandas as pd
import argparse
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crew_scheduling import config
from crew_scheduling.data_loader import DataLoader
from crew_scheduling.solvers.milp_solver import MILPSolver
from crew_scheduling.solvers.ga_solver import GASolver
from crew_scheduling.solvers.pso_solver import PSOSolver
from crew_scheduling.utils.visualization import (
    generate_all_visualizations,
    print_solution_summary
)


def export_uncovered_flights(flights: pd.DataFrame, uncovered_indices: list, 
                            output_file: str):
    """
    导出未覆盖的航班到CSV文件
    
    Args:
        flights: 航班数据框
        uncovered_indices: 未覆盖航班的索引列表
        output_file: 输出文件路径
    """
    if not uncovered_indices:
        print("所有航班都已覆盖，无需导出未覆盖航班文件")
        return
    
    uncovered = flights.iloc[uncovered_indices][
        ['FltNum', 'DptrDate', 'DptrTime', 'DptrStn', 
         'ArrvDate', 'ArrvTime', 'ArrvStn', 'Comp']
    ].copy()
    
    uncovered.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"未覆盖航班已导出到: {output_file}")


def export_crew_rosters(assignments: list, flights: pd.DataFrame, 
                        crews: pd.DataFrame, output_file: str):
    """
    导出机组排班表到CSV文件
    
    Args:
        assignments: 分配结果
        flights: 航班数据框
        crews: 机组数据框
        output_file: 输出文件路径
    """
    rows = []
    
    for assignment in assignments:
        crew_id = assignment['crew_id']
        crew_number = assignment['crew_number']
        
        # 获取该机组的所有航班
        if 'flights' in assignment:
            flight_indices = assignment['flights']
        elif 'pairings' in assignment:
            flight_indices = []
            for pairing in assignment['pairings']:
                flight_indices.extend(pairing['flights'])
        else:
            continue
        
        # 按时间排序
        flight_indices = sorted(flight_indices, 
                               key=lambda f: flights.iloc[f]['DepartureDateTime'])
        
        for flight_idx in flight_indices:
            flight = flights.iloc[flight_idx]
            
            rows.append({
                '机组编号': crew_number,
                '航班号': flight['FltNum'],
                '出发日期': flight['DptrDate'],
                '出发时间': flight['DptrTime'],
                '出发机场': flight['DptrStn'],
                '到达日期': flight['ArrvDate'],
                '到达时间': flight['ArrvTime'],
                '到达机场': flight['ArrvStn'],
                '任务类型': '执勤'  # 简化，实际应区分机长/副机长/乘机
            })
    
    roster_df = pd.DataFrame(rows)
    roster_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"机组排班表已导出到: {output_file}")


def solve_with_all_methods(loader: DataLoader, cfg: dict):
    """
    使用所有三种方法求解并比较结果
    
    Args:
        loader: 数据加载器
        cfg: 配置字典
    """
    results = {}
    
    print("\n" + "=" * 80)
    print("方法 1: MILP 精确求解")
    print("=" * 80)
    
    try:
        milp_solver = MILPSolver(loader.flights, loader.crews, cfg)
        results['MILP'] = milp_solver.solve(use_simple_model=True)
        print_solution_summary(results['MILP'], loader.flights, loader.crews)
    except Exception as e:
        print(f"MILP求解失败: {e}")
        results['MILP'] = None
    
    print("\n" + "=" * 80)
    print("方法 2: 遗传算法 (GA)")
    print("=" * 80)
    
    try:
        ga_solver = GASolver(loader.flights, loader.crews, cfg)
        results['GA'] = ga_solver.solve()
        print_solution_summary(results['GA'], loader.flights, loader.crews)
    except Exception as e:
        print(f"GA求解失败: {e}")
        results['GA'] = None
    
    print("\n" + "=" * 80)
    print("方法 3: 粒子群优化 (PSO)")
    print("=" * 80)
    
    try:
        pso_solver = PSOSolver(loader.flights, loader.crews, cfg)
        results['PSO'] = pso_solver.solve()
        print_solution_summary(results['PSO'], loader.flights, loader.crews)
    except Exception as e:
        print(f"PSO求解失败: {e}")
        results['PSO'] = None
    
    # 比较结果
    print("\n" + "=" * 80)
    print("方法比较")
    print("=" * 80)
    print(f"{'方法':<10} {'状态':<15} {'目标值':<15} {'求解时间(秒)':<15}")
    print("-" * 80)
    
    for method, result in results.items():
        if result:
            print(f"{method:<10} {result.get('status', 'N/A'):<15} "
                  f"{result.get('objective', 'N/A'):<15.2f} "
                  f"{result.get('solve_time', 'N/A'):<15.2f}")
        else:
            print(f"{method:<10} {'失败':<15} {'N/A':<15} {'N/A':<15}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='机组配对与排班优化')
    parser.add_argument('--method', type=str, default='all',
                       choices=['milp', 'ga', 'pso', 'all'],
                       help='求解方法 (milp/ga/pso/all)')
    parser.add_argument('--crew-file', type=str, default=None,
                       help='机组数据文件路径')
    parser.add_argument('--flight-file', type=str, default=None,
                       help='航班数据文件路径')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化图表')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("机组配对与排班优化系统")
    print("基于 Saemi et al., 2021 论文实现")
    print("=" * 80)
    
    # 加载配置
    cfg = {
        'MIN_CONNECTION_TIME': config.MIN_CONNECTION_TIME,
        'MAX_DAILY_FLIGHT_TIME': config.MAX_DAILY_FLIGHT_TIME,
        'MAX_DUTY_PERIOD': config.MAX_DUTY_PERIOD,
        'MIN_REST_TIME': config.MIN_REST_TIME,
        'MAX_PAIRING_DAYS': config.MAX_PAIRING_DAYS,
        'MAX_BLOCK_TIME': config.MAX_BLOCK_TIME,
        'MIN_REST_BETWEEN_DUTIES': config.MIN_REST_BETWEEN_DUTIES,
        'MAX_TAFB': config.MAX_TAFB,
        'CREW_PER_FLIGHT': config.CREW_PER_FLIGHT,
        'MILP_TIME_LIMIT': config.MILP_TIME_LIMIT,
        'MILP_GAP': config.MILP_GAP,
        'GA_POPULATION_SIZE': config.GA_POPULATION_SIZE,
        'GA_GENERATIONS': config.GA_GENERATIONS,
        'GA_CROSSOVER_RATE': config.GA_CROSSOVER_RATE,
        'GA_MUTATION_RATE': config.GA_MUTATION_RATE,
        'PSO_SWARM_SIZE': config.PSO_SWARM_SIZE,
        'PSO_ITERATIONS': config.PSO_ITERATIONS,
        'PSO_W': config.PSO_W,
        'PSO_C1': config.PSO_C1,
        'PSO_C2': config.PSO_C2,
        'DUTY_COST_WEIGHT': config.DUTY_COST_WEIGHT,
        'PAIRING_COST_WEIGHT': config.PAIRING_COST_WEIGHT,
    }
    
    # 加载数据
    print("\n加载数据...")
    crew_file = args.crew_file or config.CREW_DATA_FILE
    flight_file = args.flight_file or config.FLIGHT_DATA_FILE
    
    loader = DataLoader(crew_file, flight_file)
    crews, flights = loader.load_data()
    
    print(f"成功加载 {len(crews)} 名机组成员")
    print(f"成功加载 {len(flights)} 个航班")
    print(f"时间范围: {flights['DepartureDate'].min()} 至 {flights['DepartureDate'].max()}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 求解
    if args.method == 'all':
        results = solve_with_all_methods(loader, cfg)
        # 使用最佳结果
        best_method = min(
            [(m, r) for m, r in results.items() if r and r.get('objective')],
            key=lambda x: x[1]['objective'],
            default=(None, None)
        )
        if best_method[1]:
            result = best_method[1]
            print(f"\n最佳方法: {best_method[0]}")
        else:
            result = None
    else:
        if args.method == 'milp':
            solver = MILPSolver(loader.flights, loader.crews, cfg)
            result = solver.solve(use_simple_model=True)
        elif args.method == 'ga':
            solver = GASolver(loader.flights, loader.crews, cfg)
            result = solver.solve()
        elif args.method == 'pso':
            solver = PSOSolver(loader.flights, loader.crews, cfg)
            result = solver.solve()
        
        print_solution_summary(result, loader.flights, loader.crews)
    
    # 导出结果
    if result:
        print("\n导出结果...")
        
        # 导出机组排班表
        if result.get('assignments'):
            export_crew_rosters(
                result['assignments'],
                loader.flights,
                loader.crews,
                os.path.join(args.output_dir, config.OUTPUT_CREW_ROSTERS)
            )
        
        # 导出未覆盖航班
        uncovered = result.get('uncovered_flights', [])
        if uncovered:
            export_uncovered_flights(
                loader.flights,
                uncovered,
                os.path.join(args.output_dir, config.OUTPUT_UNCOVERED_FLIGHTS)
            )
        
        # 生成可视化
        if args.visualize:
            print("\n生成可视化图表...")
            generate_all_visualizations(
                result,
                loader.flights,
                loader.crews,
                args.output_dir
            )
    
    print("\n" + "=" * 80)
    print("程序执行完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
