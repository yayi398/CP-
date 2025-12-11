# -*- coding: utf-8 -*-
"""
遗传算法求解器
Genetic Algorithm solver for crew scheduling
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple
import pandas as pd


class GASolver:
    """遗传算法求解器"""
    
    def __init__(self, flights: pd.DataFrame, crews: pd.DataFrame, config: dict):
        """
        初始化遗传算法求解器
        
        Args:
            flights: 航班数据框
            crews: 机组数据框
            config: 配置参数
        """
        self.flights = flights
        self.crews = crews
        self.config = config
        
        self.population_size = config.get('GA_POPULATION_SIZE', 100)
        self.generations = config.get('GA_GENERATIONS', 100)
        self.crossover_rate = config.get('GA_CROSSOVER_RATE', 0.8)
        self.mutation_rate = config.get('GA_MUTATION_RATE', 0.2)
        
        self.num_flights = len(flights)
        self.num_crews = len(crews)
        
    def solve(self) -> Dict:
        """
        使用遗传算法求解
        
        Returns:
            求解结果字典
        """
        start_time = time.time()
        
        print(f"初始化种群 (大小={self.population_size})...")
        population = self._initialize_population()
        
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(self.generations):
            # 计算适应度
            fitness_scores = [self._evaluate_fitness(individual) for individual in population]
            
            # 记录最佳解
            min_fitness = min(fitness_scores)
            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_solution = population[fitness_scores.index(min_fitness)].copy()
                
            if (generation + 1) % 10 == 0:
                print(f"代数 {generation + 1}/{self.generations}, 最佳适应度: {best_fitness:.2f}")
            
            # 选择
            selected = self._selection(population, fitness_scores)
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 变异
            offspring = self._mutation(offspring)
            
            # 更新种群
            population = offspring
        
        end_time = time.time()
        
        # 构建结果
        result = {
            'status': 'Optimal' if best_solution is not None else 'Infeasible',
            'objective': best_fitness,
            'assignments': self._decode_solution(best_solution) if best_solution is not None else [],
            'solve_time': end_time - start_time,
            'algorithm': 'GA',
            'generations': self.generations
        }
        
        return result
    
    def _initialize_population(self) -> List[np.ndarray]:
        """初始化种群"""
        population = []
        
        for _ in range(self.population_size):
            # 每个个体是一个矩阵: [航班数 x 机组数]
            # individual[f, k] = 1 表示航班f分配给机组k
            individual = np.zeros((self.num_flights, self.num_crews), dtype=int)
            
            # 为每个航班随机分配机组
            for f in range(self.num_flights):
                required = self.flights.iloc[f]['TotalCrewRequired']
                # 随机选择机组
                available_crews = list(range(self.num_crews))
                selected_crews = random.sample(available_crews, min(required, len(available_crews)))
                
                for k in selected_crews:
                    individual[f, k] = 1
            
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(self, individual: np.ndarray) -> float:
        """
        评估个体的适应度（成本越低越好）
        包含工作负载平衡因子
        
        Args:
            individual: 个体（分配矩阵）
            
        Returns:
            适应度值（成本）
        """
        total_cost = 0
        penalty = 0
        
        # 计算执勤成本
        for f in range(self.num_flights):
            flight = self.flights.iloc[f]
            flight_hours = flight['FlightDuration'] / 60.0
            
            for k in range(self.num_crews):
                if individual[f, k] == 1:
                    crew = self.crews.iloc[k]
                    total_cost += crew['DutyCostPerHour'] * flight_hours
        
        # 惩罚项1: 航班覆盖不足或过度
        for f in range(self.num_flights):
            required = self.flights.iloc[f]['TotalCrewRequired']
            assigned = individual[f, :].sum()
            
            if assigned != required:
                penalty += abs(assigned - required) * 10000  # 大惩罚
        
        # 惩罚项2: 机组时间冲突
        # Note: O(n²) complexity for each crew. For large datasets, consider optimizing with interval trees
        crew_hours = np.zeros(self.num_crews)
        for k in range(self.num_crews):
            assigned_flights = np.where(individual[:, k] == 1)[0]
            
            # 计算工作时长（用于平衡）
            for f in assigned_flights:
                crew_hours[k] += self.flights.iloc[f]['FlightDuration'] / 60.0
            
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
                        penalty += 5000  # 时间冲突惩罚
        
        # 惩罚项3: 工作负载不平衡（添加标准差惩罚）
        if crew_hours.sum() > 0:
            workload_std = np.std(crew_hours)
            penalty += workload_std * 100  # 工作负载不平衡惩罚
        
        return total_cost + penalty
    
    def _selection(self, population: List[np.ndarray], 
                   fitness_scores: List[float]) -> List[np.ndarray]:
        """
        选择操作（锦标赛选择）
        
        Args:
            population: 种群
            fitness_scores: 适应度分数
            
        Returns:
            选择后的个体列表
        """
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # 锦标赛选择
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """
        交叉操作（单点交叉）
        
        Args:
            population: 种群
            
        Returns:
            交叉后的后代
        """
        offspring = []
        
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]
            
            if random.random() < self.crossover_rate:
                # 单点交叉
                crossover_point = random.randint(1, self.num_flights - 1)
                
                child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parent1.copy())
                offspring.append(parent2.copy())
        
        return offspring[:len(population)]
    
    def _mutation(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """
        变异操作
        
        Args:
            population: 种群
            
        Returns:
            变异后的种群
        """
        for individual in population:
            if random.random() < self.mutation_rate:
                # 随机选择一个航班
                f = random.randint(0, self.num_flights - 1)
                
                # 重新分配机组
                required = self.flights.iloc[f]['TotalCrewRequired']
                individual[f, :] = 0  # 清除原分配
                
                available_crews = list(range(self.num_crews))
                selected_crews = random.sample(available_crews, min(required, len(available_crews)))
                
                for k in selected_crews:
                    individual[f, k] = 1
        
        return population
    
    def _decode_solution(self, individual: np.ndarray) -> List[Dict]:
        """
        解码解（转换为分配列表）
        
        Args:
            individual: 个体
            
        Returns:
            分配列表
        """
        assignments = []
        
        for k in range(self.num_crews):
            flights = np.where(individual[:, k] == 1)[0].tolist()
            
            if flights:
                assignments.append({
                    'crew_id': k,
                    'crew_number': self.crews.iloc[k]['EmpNo'],
                    'flights': flights
                })
        
        return assignments


def solve_crew_scheduling_ga(flights: pd.DataFrame, crews: pd.DataFrame, 
                             config: dict) -> Dict:
    """
    便捷函数：使用遗传算法求解机组排班问题
    
    Args:
        flights: 航班数据框
        crews: 机组数据框
        config: 配置参数
        
    Returns:
        求解结果
    """
    solver = GASolver(flights, crews, config)
    return solver.solve()
