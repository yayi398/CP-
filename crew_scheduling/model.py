# -*- coding: utf-8 -*-
"""
数学模型定义
Mathematical model for crew pairing and rostering
基于 Saemi et al., 2021 论文
"""

import pulp
import pandas as pd
from typing import Dict, List, Tuple, Set
from datetime import datetime, timedelta


class CrewPairingModel:
    """机组配对与排班集成优化模型 (MILP)"""
    
    def __init__(self, flights: pd.DataFrame, crews: pd.DataFrame, 
                 pairings: List[Dict], config: dict):
        """
        初始化模型
        
        Args:
            flights: 航班数据框
            crews: 机组数据框
            pairings: 可行配对列表
            config: 配置参数
        """
        self.flights = flights
        self.crews = crews
        self.pairings = pairings
        self.config = config
        
        # 创建索引集合
        self.F = list(range(len(flights)))  # 航班集合
        self.K = list(range(len(crews)))    # 机组集合
        self.P = list(range(len(pairings))) # 配对集合
        
        # 决策变量
        self.x = {}  # x[k,p]: 机组k是否被分配到配对p
        self.y = {}  # y[f,p]: 航班f是否包含在配对p中
        self.z = {}  # z[f,k]: 机组k是否执行航班f
        
        # PuLP问题
        self.prob = None
        
    def build_model(self):
        """构建MILP模型"""
        # 创建问题
        self.prob = pulp.LpProblem("Crew_Pairing_Rostering", pulp.LpMinimize)
        
        # 定义决策变量
        self._define_variables()
        
        # 定义目标函数
        self._define_objective()
        
        # 定义约束条件
        self._define_constraints()
        
    def _define_variables(self):
        """定义决策变量"""
        # x[k,p]: 机组k是否被分配到配对p
        for k in self.K:
            for p in self.P:
                self.x[k, p] = pulp.LpVariable(f"x_{k}_{p}", cat='Binary')
        
        # y[f,p]: 航班f是否包含在配对p中
        for f in self.F:
            for p in self.P:
                # 检查航班f是否在配对p中
                if f in self.pairings[p]['flights']:
                    self.y[f, p] = 1  # 固定为1（航班在配对中）
                else:
                    self.y[f, p] = 0  # 固定为0（航班不在配对中）
        
        # z[f,k]: 机组k是否执行航班f（派生变量，用于某些约束）
        for f in self.F:
            for k in self.K:
                self.z[f, k] = pulp.LpVariable(f"z_{f}_{k}", cat='Binary')
    
    def _define_objective(self):
        """定义目标函数：最小化总成本"""
        # 成本组成：执勤成本 + 配对成本
        
        duty_cost = 0  # 执勤成本
        pairing_cost = 0  # 配对成本
        
        for k in self.K:
            crew = self.crews.iloc[k]
            duty_cost_rate = crew['DutyCostPerHour']
            pairing_cost_rate = crew['ParingCostPerHour']
            
            for p in self.P:
                pairing = self.pairings[p]
                # 执勤时间（小时）
                duty_hours = pairing['total_duration'] / 60.0
                # 配对时间（小时）
                pairing_hours = pairing['total_duration'] / 60.0
                
                duty_cost += self.x[k, p] * duty_cost_rate * duty_hours
                pairing_cost += self.x[k, p] * pairing_cost_rate * pairing_hours
        
        # 总目标函数
        self.prob += (
            self.config.get('DUTY_COST_WEIGHT', 1.0) * duty_cost +
            self.config.get('PAIRING_COST_WEIGHT', 1.0) * pairing_cost
        )
    
    def _define_constraints(self):
        """定义约束条件"""
        # 1. 航班覆盖约束：每个航班必须被恰好覆盖所需次数
        self._add_flight_coverage_constraints()
        
        # 2. 机组可用性约束：每个机组在同一时间只能执行一个任务
        self._add_crew_availability_constraints()
        
        # 3. 配对连续性约束（已在配对生成时保证）
        
        # 4. 基地约束（已在配对生成时保证）
        
        # 5. 链接约束：z变量与x、y的关系
        self._add_linking_constraints()
        
    def _add_flight_coverage_constraints(self):
        """航班覆盖约束：每个航班需要指定数量的机组"""
        for f in self.F:
            flight = self.flights.iloc[f]
            required_crew = flight['TotalCrewRequired']
            
            # 对每个航班，所有包含它的配对分配的机组总数应等于需求
            self.prob += (
                pulp.lpSum(self.x[k, p] * self.y[f, p] 
                          for k in self.K for p in self.P) == required_crew,
                f"FlightCoverage_{f}"
            )
    
    def _add_crew_availability_constraints(self):
        """机组可用性约束：每个机组每天最多一个执勤"""
        # 获取所有日期
        dates = sorted(self.flights['DepartureDate'].unique())
        
        for k in self.K:
            # 每个机组最多被分配到一个配对（简化版本）
            # 在实际应用中，应该考虑时间窗口
            for d in dates:
                # 获取该日期的所有配对
                pairings_on_date = []
                for p in self.P:
                    pairing = self.pairings[p]
                    # 检查配对是否在该日期有航班
                    for flight_idx in pairing['flights']:
                        flight = self.flights.iloc[flight_idx]
                        if flight['DepartureDate'] == d:
                            pairings_on_date.append(p)
                            break
                
                # 每个机组在每天最多一个执勤
                if pairings_on_date:
                    self.prob += (
                        pulp.lpSum(self.x[k, p] for p in pairings_on_date) <= 1,
                        f"CrewAvail_{k}_{d}"
                    )
    
    def _add_linking_constraints(self):
        """链接约束：z[f,k] = sum over p of x[k,p] * y[f,p]"""
        for f in self.F:
            for k in self.K:
                self.prob += (
                    self.z[f, k] == pulp.lpSum(self.x[k, p] * self.y[f, p] for p in self.P),
                    f"Link_{f}_{k}"
                )
    
    def solve(self, time_limit: int = 300, gap: float = 0.05) -> Dict:
        """
        求解模型
        
        Args:
            time_limit: 时间限制（秒）
            gap: MIP gap容忍度
            
        Returns:
            求解结果字典
        """
        if self.prob is None:
            self.build_model()
        
        # 使用CBC求解器
        solver = pulp.PULP_CBC_CMD(
            timeLimit=time_limit,
            gapRel=gap,
            msg=1
        )
        
        # 求解
        status = self.prob.solve(solver)
        
        # 提取结果
        result = {
            'status': pulp.LpStatus[status],
            'objective': pulp.value(self.prob.objective),
            'assignments': self._extract_assignments(),
            'uncovered_flights': self._get_uncovered_flights()
        }
        
        return result
    
    def _extract_assignments(self) -> List[Dict]:
        """提取机组分配结果"""
        assignments = []
        
        for k in self.K:
            crew = self.crews.iloc[k]
            crew_assignments = []
            
            for p in self.P:
                if pulp.value(self.x[k, p]) > 0.5:  # 被分配
                    pairing = self.pairings[p]
                    crew_assignments.append({
                        'pairing_id': p,
                        'flights': pairing['flights'],
                        'start_time': pairing['start_time'],
                        'end_time': pairing['end_time']
                    })
            
            if crew_assignments:
                assignments.append({
                    'crew_id': k,
                    'crew_number': crew['EmpNo'],
                    'pairings': crew_assignments
                })
        
        return assignments
    
    def _get_uncovered_flights(self) -> List[int]:
        """获取未覆盖的航班列表"""
        uncovered = []
        
        for f in self.F:
            flight = self.flights.iloc[f]
            required = flight['TotalCrewRequired']
            
            # 计算实际分配的机组数
            assigned = sum(
                pulp.value(self.x[k, p]) * self.y[f, p]
                for k in self.K for p in self.P
            )
            
            if assigned < required:
                uncovered.append(f)
        
        return uncovered


def create_simple_model(flights: pd.DataFrame, crews: pd.DataFrame, config: dict):
    """
    创建简化模型（直接分配航班到机组，不通过配对）
    用于快速测试和小规模问题
    
    Args:
        flights: 航班数据框
        crews: 机组数据框
        config: 配置参数
        
    Returns:
        PuLP问题对象
    """
    # 创建问题
    prob = pulp.LpProblem("Simple_Crew_Assignment", pulp.LpMinimize)
    
    # 决策变量: x[f,k] = 机组k是否执行航班f
    F = list(range(len(flights)))
    K = list(range(len(crews)))
    
    x = {}
    for f in F:
        for k in K:
            x[f, k] = pulp.LpVariable(f"x_{f}_{k}", cat='Binary')
    
    # 目标函数：最小化成本
    cost = 0
    for f in F:
        for k in K:
            flight = flights.iloc[f]
            crew = crews.iloc[k]
            flight_hours = flight['FlightDuration'] / 60.0
            cost += x[f, k] * crew['DutyCostPerHour'] * flight_hours
    
    prob += cost
    
    # 约束1：每个航班需要指定数量的机组
    for f in F:
        flight = flights.iloc[f]
        prob += (
            pulp.lpSum(x[f, k] for k in K) == flight['TotalCrewRequired'],
            f"Flight_{f}"
        )
    
    # 约束2：机组不能同时执行两个航班（时间冲突检测）
    for k in K:
        for f1 in F:
            for f2 in F:
                if f1 < f2:
                    flight1 = flights.iloc[f1]
                    flight2 = flights.iloc[f2]
                    
                    # 检查时间是否重叠
                    if not (flight1['ArrivalDateTime'] <= flight2['DepartureDateTime'] or
                            flight2['ArrivalDateTime'] <= flight1['DepartureDateTime']):
                        # 有重叠，不能同时分配
                        prob += (
                            x[f1, k] + x[f2, k] <= 1,
                            f"Conflict_{k}_{f1}_{f2}"
                        )
    
    return prob, x
