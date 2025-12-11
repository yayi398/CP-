# -*- coding: utf-8 -*-
"""
MILP精确求解器
MILP exact solver using PuLP + CBC
"""

import time
from typing import Dict, List
import pandas as pd
from crew_scheduling.model import CrewPairingModel, create_simple_model
from crew_scheduling.utils.pairing_generator import PairingGenerator, generate_simple_pairings


class MILPSolver:
    """MILP精确求解器"""
    
    def __init__(self, flights: pd.DataFrame, crews: pd.DataFrame, config: dict):
        """
        初始化求解器
        
        Args:
            flights: 航班数据框
            crews: 机组数据框
            config: 配置参数
        """
        self.flights = flights
        self.crews = crews
        self.config = config
        self.model = None
        self.solution = None
        
    def solve(self, use_simple_model: bool = False) -> Dict:
        """
        求解机组排班问题
        
        Args:
            use_simple_model: 是否使用简化模型（直接分配，不生成配对）
            
        Returns:
            求解结果字典
        """
        start_time = time.time()
        
        if use_simple_model:
            result = self._solve_simple_model()
        else:
            result = self._solve_with_pairings()
        
        end_time = time.time()
        result['solve_time'] = end_time - start_time
        
        return result
    
    def _solve_simple_model(self) -> Dict:
        """使用简化模型求解（不生成配对）"""
        print("使用简化模型求解...")
        
        # 创建简化模型
        prob, x = create_simple_model(self.flights, self.crews, self.config)
        
        # 求解
        import pulp
        solver = pulp.PULP_CBC_CMD(
            timeLimit=self.config.get('MILP_TIME_LIMIT', 300),
            gapRel=self.config.get('MILP_GAP', 0.05),
            msg=1
        )
        
        status = prob.solve(solver)
        
        # 提取结果
        result = {
            'status': pulp.LpStatus[status],
            'objective': pulp.value(prob.objective) if status == 1 else None,
            'model_type': 'simple',
            'assignments': []
        }
        
        # 提取分配结果
        if status == 1:  # 最优解
            for k in range(len(self.crews)):
                crew_flights = []
                for f in range(len(self.flights)):
                    if pulp.value(x[f, k]) > 0.5:
                        crew_flights.append(f)
                
                if crew_flights:
                    result['assignments'].append({
                        'crew_id': k,
                        'crew_number': self.crews.iloc[k]['EmpNo'],
                        'flights': crew_flights
                    })
        
        return result
    
    def _solve_with_pairings(self) -> Dict:
        """使用配对生成方法求解"""
        print("生成可行配对...")
        
        # 获取基地
        base = self.crews.iloc[0]['Base']
        
        # 生成简单的往返配对
        pairings_list = generate_simple_pairings(
            self.flights,
            base,
            self.config.get('MIN_CONNECTION_TIME', 40),
            self.config.get('MAX_PAIRING_DAYS', 4)
        )
        
        # 如果没有生成配对，使用简化模型
        if len(pairings_list) == 0:
            print("未生成任何配对，回退到简化模型...")
            return self._solve_simple_model()
        
        print(f"生成了 {len(pairings_list)} 个配对")
        
        # 转换为配对信息字典格式
        pairings = []
        for pairing_flights in pairings_list:
            if len(pairing_flights) > 0:
                start_time = self.flights.iloc[pairing_flights[0]]['DepartureDateTime']
                end_time = self.flights.iloc[pairing_flights[-1]]['ArrivalDateTime']
                total_duration = (end_time - start_time).total_seconds() / 60
                
                pairings.append({
                    'flights': pairing_flights,
                    'num_flights': len(pairing_flights),
                    'start_time': start_time,
                    'end_time': end_time,
                    'total_duration': total_duration,
                    'total_flight_time': sum(self.flights.iloc[f]['FlightDuration'] 
                                            for f in pairing_flights),
                    'start_station': self.flights.iloc[pairing_flights[0]]['DptrStn'],
                    'end_station': self.flights.iloc[pairing_flights[-1]]['ArrvStn']
                })
        
        print(f"创建模型，使用 {len(pairings)} 个配对...")
        
        # 创建并求解模型
        self.model = CrewPairingModel(self.flights, self.crews, pairings, self.config)
        self.model.build_model()
        
        print("求解模型...")
        result = self.model.solve(
            time_limit=self.config.get('MILP_TIME_LIMIT', 300),
            gap=self.config.get('MILP_GAP', 0.05)
        )
        
        result['model_type'] = 'pairing'
        result['num_pairings'] = len(pairings)
        
        return result
    
    def get_statistics(self, result: Dict) -> Dict:
        """
        计算求解统计信息
        
        Args:
            result: 求解结果
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_flights': len(self.flights),
            'total_crews': len(self.crews),
            'status': result.get('status', 'Unknown'),
            'objective_value': result.get('objective', None),
            'solve_time_seconds': result.get('solve_time', None),
            'model_type': result.get('model_type', 'unknown')
        }
        
        # 计算覆盖的航班数
        if 'assignments' in result:
            covered_flights = set()
            for assignment in result['assignments']:
                if 'flights' in assignment:
                    covered_flights.update(assignment['flights'])
                elif 'pairings' in assignment:
                    for pairing in assignment['pairings']:
                        covered_flights.update(pairing['flights'])
            
            stats['covered_flights'] = len(covered_flights)
            stats['uncovered_flights'] = len(self.flights) - len(covered_flights)
        
        # 计算未覆盖航班
        if 'uncovered_flights' in result:
            stats['uncovered_flights'] = len(result['uncovered_flights'])
        
        return stats


def solve_crew_scheduling_milp(flights: pd.DataFrame, crews: pd.DataFrame, 
                               config: dict, use_simple: bool = False) -> Dict:
    """
    便捷函数：使用MILP求解机组排班问题
    
    Args:
        flights: 航班数据框
        crews: 机组数据框
        config: 配置参数
        use_simple: 是否使用简化模型
        
    Returns:
        求解结果
    """
    solver = MILPSolver(flights, crews, config)
    result = solver.solve(use_simple_model=use_simple)
    stats = solver.get_statistics(result)
    result['statistics'] = stats
    return result
