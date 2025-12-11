# -*- coding: utf-8 -*-
"""
配对生成器
Pairing generator - generates feasible crew pairings
"""

from typing import List, Dict, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
from itertools import combinations


class PairingGenerator:
    """生成可行的机组配对（Pairing）"""
    
    def __init__(self, flights: pd.DataFrame, config: dict):
        """
        初始化配对生成器
        
        Args:
            flights: 航班数据框
            config: 配置参数字典
        """
        self.flights = flights
        self.config = config
        self.pairings = []
        
    def generate_pairings(self, base: str, max_pairings: int = None) -> List[Dict]:
        """
        生成从指定基地出发的所有可行配对
        
        Args:
            base: 基地机场代码
            max_pairings: 最大配对数量（可选，用于限制）
            
        Returns:
            配对列表，每个配对包含航班序列和相关信息
        """
        self.pairings = []
        
        # 获取从基地出发的航班
        base_departures = self.flights[self.flights['DptrStn'] == base].index.tolist()
        
        for start_flight in base_departures:
            # 从每个基地出发航班开始，生成配对
            self._build_pairing_recursive(
                base=base,
                current_pairing=[start_flight],
                current_location=self.flights.loc[start_flight, 'ArrvStn'],
                current_time=self.flights.loc[start_flight, 'ArrivalDateTime'],
                pairing_start_time=self.flights.loc[start_flight, 'DepartureDateTime'],
                duty_flights=[start_flight],
                duty_start_time=self.flights.loc[start_flight, 'DepartureDateTime']
            )
            
            if max_pairings and len(self.pairings) >= max_pairings:
                break
        
        return self.pairings
    
    def _build_pairing_recursive(self, base: str, current_pairing: List[int],
                                 current_location: str, current_time: datetime,
                                 pairing_start_time: datetime,
                                 duty_flights: List[int],
                                 duty_start_time: datetime):
        """
        递归构建配对
        
        Args:
            base: 基地
            current_pairing: 当前配对中的航班列表
            current_location: 当前位置
            current_time: 当前时间
            pairing_start_time: 配对开始时间
            duty_flights: 当前执勤中的航班
            duty_start_time: 当前执勤开始时间
        """
        # 检查是否回到基地
        if current_location == base and len(current_pairing) > 1:
            # 形成一个完整的配对
            pairing_info = self._create_pairing_info(current_pairing, pairing_start_time, current_time)
            if self._is_valid_pairing(pairing_info):
                self.pairings.append(pairing_info)
        
        # 检查配对持续时间是否超过最大限制
        pairing_duration_days = (current_time - pairing_start_time).total_seconds() / (24 * 3600)
        if pairing_duration_days > self.config.get('MAX_PAIRING_DAYS', 4):
            return
        
        # 查找下一个可能的航班
        next_flights = self._find_next_flights(current_location, current_time, current_pairing)
        
        for next_flight in next_flights:
            next_flight_info = self.flights.loc[next_flight]
            time_diff = (next_flight_info['DepartureDateTime'] - current_time).total_seconds() / 60
            
            # 检查是否需要开始新的执勤
            if time_diff >= self.config.get('MIN_REST_BETWEEN_DUTIES', 600):
                # 开始新的执勤
                new_duty_flights = [next_flight]
                new_duty_start_time = next_flight_info['DepartureDateTime']
            else:
                # 继续当前执勤
                new_duty_flights = duty_flights + [next_flight]
                new_duty_start_time = duty_start_time
                
                # 检查执勤时间约束
                duty_duration = (next_flight_info['ArrivalDateTime'] - duty_start_time).total_seconds() / 60
                if duty_duration > self.config.get('MAX_DUTY_PERIOD', 720):
                    continue
                
                # 检查执勤飞行时间约束
                duty_flight_time = sum(self.flights.loc[f, 'FlightDuration'] for f in new_duty_flights)
                if duty_flight_time > self.config.get('MAX_BLOCK_TIME', 600):
                    continue
            
            # 递归继续构建
            self._build_pairing_recursive(
                base=base,
                current_pairing=current_pairing + [next_flight],
                current_location=next_flight_info['ArrvStn'],
                current_time=next_flight_info['ArrivalDateTime'],
                pairing_start_time=pairing_start_time,
                duty_flights=new_duty_flights,
                duty_start_time=new_duty_start_time
            )
    
    def _find_next_flights(self, location: str, current_time: datetime, 
                          current_pairing: List[int]) -> List[int]:
        """
        查找从当前位置和时间可以连接的下一个航班
        
        Args:
            location: 当前位置
            current_time: 当前时间
            current_pairing: 当前配对（避免重复）
            
        Returns:
            可连接的航班索引列表
        """
        next_flights = []
        
        # 从当前位置出发的航班
        candidates = self.flights[
            (self.flights['DptrStn'] == location) & 
            (self.flights['DepartureDateTime'] > current_time)
        ]
        
        for idx, flight in candidates.iterrows():
            if idx in current_pairing:
                continue
                
            time_diff = (flight['DepartureDateTime'] - current_time).total_seconds() / 60
            
            # 检查最小连接时间
            if time_diff >= self.config.get('MIN_CONNECTION_TIME', 40):
                next_flights.append(idx)
        
        return next_flights
    
    def _create_pairing_info(self, flight_indices: List[int], 
                            start_time: datetime, end_time: datetime) -> Dict:
        """
        创建配对信息字典
        
        Args:
            flight_indices: 航班索引列表
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            配对信息字典
        """
        flights_info = self.flights.loc[flight_indices]
        
        total_flight_time = flights_info['FlightDuration'].sum()
        total_duration = (end_time - start_time).total_seconds() / 60
        
        return {
            'flights': flight_indices,
            'num_flights': len(flight_indices),
            'start_time': start_time,
            'end_time': end_time,
            'total_duration': total_duration,
            'total_flight_time': total_flight_time,
            'start_station': flights_info.iloc[0]['DptrStn'],
            'end_station': flights_info.iloc[-1]['ArrvStn']
        }
    
    def _is_valid_pairing(self, pairing_info: Dict) -> bool:
        """
        检查配对是否有效
        
        Args:
            pairing_info: 配对信息字典
            
        Returns:
            是否有效
        """
        # 必须从基地开始并返回基地
        if pairing_info['start_station'] != pairing_info['end_station']:
            return False
        
        # 检查时长约束
        max_tafb = self.config.get('MAX_TAFB', 14400)
        if pairing_info['total_duration'] > max_tafb:
            return False
        
        return True


def generate_simple_pairings(flights: pd.DataFrame, base: str, 
                            min_connection: int, max_pairing_days: int) -> List[List[int]]:
    """
    生成简单的往返配对（用于快速测试）
    
    Args:
        flights: 航班数据框
        base: 基地
        min_connection: 最小连接时间
        max_pairing_days: 最大配对天数
        
    Returns:
        配对列表（每个配对是航班索引列表）
    """
    simple_pairings = []
    
    # 获取从基地出发的航班
    outbound = flights[flights['DptrStn'] == base]
    
    for out_idx, out_flight in outbound.iterrows():
        # 查找可以返回基地的航班
        inbound = flights[
            (flights['DptrStn'] == out_flight['ArrvStn']) &
            (flights['ArrvStn'] == base) &
            (flights['DepartureDateTime'] > out_flight['ArrivalDateTime'])
        ]
        
        for in_idx, in_flight in inbound.iterrows():
            time_diff = (in_flight['DepartureDateTime'] - out_flight['ArrivalDateTime']).total_seconds() / 60
            
            if time_diff >= min_connection:
                # 检查总时长
                total_duration = (in_flight['ArrivalDateTime'] - out_flight['DepartureDateTime']).total_seconds()
                if total_duration / (24 * 3600) <= max_pairing_days:
                    simple_pairings.append([out_idx, in_idx])
    
    return simple_pairings
