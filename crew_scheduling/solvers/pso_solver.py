# -*- coding: utf-8 -*-
"""
粒子群优化求解器
Particle Swarm Optimization solver for crew scheduling
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple
import pandas as pd


class PSOSolver:
    """粒子群优化求解器"""
    
    def __init__(self, flights: pd.DataFrame, crews: pd.DataFrame, config: dict):
        """
        初始化PSO求解器
        
        Args:
            flights: 航班数据框
            crews: 机组数据框
            config: 配置参数
        """
        self.flights = flights
        self.crews = crews
        self.config = config
        
        self.swarm_size = config.get('PSO_SWARM_SIZE', 50)
        self.iterations = config.get('PSO_ITERATIONS', 100)
        self.w = config.get('PSO_W', 0.7)  # 惯性权重
        self.c1 = config.get('PSO_C1', 1.5)  # 个体学习因子
        self.c2 = config.get('PSO_C2', 1.5)  # 社会学习因子
        
        self.num_flights = len(flights)
        self.num_crews = len(crews)
        self.dimension = self.num_flights * self.num_crews
        
    def solve(self) -> Dict:
        """
        使用粒子群优化求解
        
        Returns:
            求解结果字典
        """
        start_time = time.time()
        
        print(f"初始化粒子群 (大小={self.swarm_size})...")
        
        # 初始化粒子位置和速度
        particles = self._initialize_swarm()
        velocities = np.random.randn(self.swarm_size, self.dimension) * 0.1
        
        # 个体最佳位置和适应度
        pbest_positions = particles.copy()
        pbest_fitness = np.array([self._evaluate_fitness(p) for p in particles])
        
        # 全局最佳位置和适应度
        gbest_idx = np.argmin(pbest_fitness)
        gbest_position = particles[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]
        
        # 迭代优化
        for iteration in range(self.iterations):
            for i in range(self.swarm_size):
                # 更新速度
                r1 = np.random.rand(self.dimension)
                r2 = np.random.rand(self.dimension)
                
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (pbest_positions[i] - particles[i]) +
                    self.c2 * r2 * (gbest_position - particles[i])
                )
                
                # 限制速度
                velocities[i] = np.clip(velocities[i], -1.0, 1.0)
                
                # 更新位置
                particles[i] = particles[i] + velocities[i]
                
                # 应用约束（转换为有效的分配）
                particles[i] = self._apply_constraints(particles[i])
                
                # 评估适应度
                fitness = self._evaluate_fitness(particles[i])
                
                # 更新个体最佳
                if fitness < pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = particles[i].copy()
                    
                    # 更新全局最佳
                    if fitness < gbest_fitness:
                        gbest_fitness = fitness
                        gbest_position = particles[i].copy()
            
            if (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration + 1}/{self.iterations}, 最佳适应度: {gbest_fitness:.2f}")
        
        end_time = time.time()
        
        # 构建结果
        result = {
            'status': 'Optimal',
            'objective': gbest_fitness,
            'assignments': self._decode_solution(gbest_position),
            'solve_time': end_time - start_time,
            'algorithm': 'PSO',
            'iterations': self.iterations
        }
        
        return result
    
    def _initialize_swarm(self) -> np.ndarray:
        """
        初始化粒子群
        
        Returns:
            粒子位置矩阵
        """
        swarm = []
        
        for _ in range(self.swarm_size):
            # 创建随机解
            particle = np.random.rand(self.dimension)
            # 应用约束
            particle = self._apply_constraints(particle)
            swarm.append(particle)
        
        return np.array(swarm)
    
    def _apply_constraints(self, particle: np.ndarray) -> np.ndarray:
        """
        应用约束，将连续解转换为离散分配
        
        Args:
            particle: 粒子位置（连续值）
            
        Returns:
            调整后的粒子位置
        """
        # 将粒子重塑为 [航班数 x 机组数] 矩阵
        assignment = particle.reshape(self.num_flights, self.num_crews)
        
        # 对每个航班，选择概率最高的机组
        constrained = np.zeros_like(assignment)
        
        for f in range(self.num_flights):
            required = self.flights.iloc[f]['TotalCrewRequired']
            
            # 获取概率排序（sigmoid转换）
            probs = 1 / (1 + np.exp(-assignment[f]))
            
            # 选择概率最高的机组
            top_crews = np.argsort(probs)[-required:]
            constrained[f, top_crews] = 1.0
        
        return constrained.flatten()
    
    def _evaluate_fitness(self, particle: np.ndarray) -> float:
        """
        评估粒子的适应度
        
        Args:
            particle: 粒子位置
            
        Returns:
            适应度值（成本）
        """
        # 将粒子重塑为分配矩阵
        assignment = particle.reshape(self.num_flights, self.num_crews)
        
        total_cost = 0
        penalty = 0
        
        # 计算执勤成本
        for f in range(self.num_flights):
            flight = self.flights.iloc[f]
            flight_hours = flight['FlightDuration'] / 60.0
            
            for k in range(self.num_crews):
                if assignment[f, k] > 0.5:  # 被分配
                    crew = self.crews.iloc[k]
                    total_cost += crew['DutyCostPerHour'] * flight_hours
        
        # 惩罚项1: 航班覆盖不足或过度
        for f in range(self.num_flights):
            required = self.flights.iloc[f]['TotalCrewRequired']
            assigned = (assignment[f, :] > 0.5).sum()
            
            if assigned != required:
                penalty += abs(assigned - required) * 10000
        
        # 惩罚项2: 机组时间冲突
        for k in range(self.num_crews):
            assigned_flights = np.where(assignment[:, k] > 0.5)[0]
            
            # 检查时间冲突
            for i in range(len(assigned_flights)):
                for j in range(i + 1, len(assigned_flights)):
                    f1 = assigned_flights[i]
                    f2 = assigned_flights[j]
                    
                    flight1 = self.flights.iloc[f1]
                    flight2 = self.flights.iloc[f2]
                    
                    # 检查时间重叠
                    if not (flight1['ArrivalDateTime'] <= flight2['DepartureDateTime'] or
                           flight2['ArrivalDateTime'] <= flight1['DepartureDateTime']):
                        penalty += 5000
        
        return total_cost + penalty
    
    def _decode_solution(self, particle: np.ndarray) -> List[Dict]:
        """
        解码解（转换为分配列表）
        
        Args:
            particle: 粒子位置
            
        Returns:
            分配列表
        """
        assignment = particle.reshape(self.num_flights, self.num_crews)
        assignments = []
        
        for k in range(self.num_crews):
            flights = np.where(assignment[:, k] > 0.5)[0].tolist()
            
            if flights:
                assignments.append({
                    'crew_id': k,
                    'crew_number': self.crews.iloc[k]['EmpNo'],
                    'flights': flights
                })
        
        return assignments


def solve_crew_scheduling_pso(flights: pd.DataFrame, crews: pd.DataFrame, 
                              config: dict) -> Dict:
    """
    便捷函数：使用粒子群优化求解机组排班问题
    
    Args:
        flights: 航班数据框
        crews: 机组数据框
        config: 配置参数
        
    Returns:
        求解结果
    """
    solver = PSOSolver(flights, crews, config)
    return solver.solve()
