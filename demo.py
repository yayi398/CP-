#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单演示脚本
Quick demo of the crew scheduling optimization system
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crew_scheduling.data_loader import DataLoader
from crew_scheduling.solvers.ga_solver import GASolver
from crew_scheduling.solvers.pso_solver import PSOSolver
from crew_scheduling import config


def main():
    """运行简单演示"""
    print("=" * 80)
    print("机组配对与排班优化 - 快速演示")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    loader = DataLoader(config.CREW_DATA_FILE, config.FLIGHT_DATA_FILE)
    crews, flights = loader.load_data()
    
    print(f"  ✓ 成功加载 {len(crews)} 名机组成员")
    print(f"  ✓ 成功加载 {len(flights)} 个航班")
    print(f"  ✓ 时间范围: {flights['DepartureDate'].min()} 至 {flights['DepartureDate'].max()}")
    
    # 使用前50个航班进行快速演示
    flights_subset = flights[:50]
    print(f"  ✓ 使用前 {len(flights_subset)} 个航班进行演示")
    
    # 2. 配置参数
    print("\n[2/4] 配置参数...")
    demo_config = {
        'GA_POPULATION_SIZE': 30,
        'GA_GENERATIONS': 20,
        'GA_CROSSOVER_RATE': 0.8,
        'GA_MUTATION_RATE': 0.2,
        'PSO_SWARM_SIZE': 30,
        'PSO_ITERATIONS': 20,
        'PSO_W': 0.7,
        'PSO_C1': 1.5,
        'PSO_C2': 1.5,
    }
    print("  ✓ 参数配置完成")
    
    # 3. 使用遗传算法求解
    print("\n[3/4] 使用遗传算法求解...")
    ga_solver = GASolver(flights_subset, crews, demo_config)
    ga_result = ga_solver.solve()
    
    print(f"  ✓ GA求解完成")
    print(f"    - 状态: {ga_result['status']}")
    print(f"    - 目标值: {ga_result['objective']:.2f}")
    print(f"    - 求解时间: {ga_result['solve_time']:.2f} 秒")
    print(f"    - 分配机组数: {len(ga_result['assignments'])}")
    
    # 4. 使用粒子群优化求解
    print("\n[4/4] 使用粒子群优化求解...")
    pso_solver = PSOSolver(flights_subset, crews, demo_config)
    pso_result = pso_solver.solve()
    
    print(f"  ✓ PSO求解完成")
    print(f"    - 状态: {pso_result['status']}")
    print(f"    - 目标值: {pso_result['objective']:.2f}")
    print(f"    - 求解时间: {pso_result['solve_time']:.2f} 秒")
    print(f"    - 分配机组数: {len(pso_result['assignments'])}")
    
    # 5. 比较结果
    print("\n" + "=" * 80)
    print("结果比较")
    print("=" * 80)
    print(f"{'算法':<15} {'目标值':<15} {'求解时间(秒)':<15} {'相对差异':<15}")
    print("-" * 80)
    
    best_obj = min(ga_result['objective'], pso_result['objective'])
    
    ga_diff = (ga_result['objective'] - best_obj) / best_obj * 100
    pso_diff = (pso_result['objective'] - best_obj) / best_obj * 100
    
    print(f"{'遗传算法':<15} {ga_result['objective']:<15.2f} "
          f"{ga_result['solve_time']:<15.2f} {ga_diff:<15.2f}%")
    print(f"{'粒子群优化':<15} {pso_result['objective']:<15.2f} "
          f"{pso_result['solve_time']:<15.2f} {pso_diff:<15.2f}%")
    
    print("\n" + "=" * 80)
    if pso_result['objective'] < ga_result['objective']:
        print("✓ PSO找到了更好的解!")
    elif ga_result['objective'] < pso_result['objective']:
        print("✓ GA找到了更好的解!")
    else:
        print("✓ 两种算法找到了相同质量的解!")
    print("=" * 80)
    
    print("\n提示: 运行完整程序请使用:")
    print("  python -m crew_scheduling.main --method all --visualize")


if __name__ == '__main__':
    main()
