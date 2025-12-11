# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
Data loading and preprocessing module
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, crew_file: str, flight_file: str):
        """
        初始化数据加载器
        
        Args:
            crew_file: 机组数据文件路径
            flight_file: 航班数据文件路径
        """
        self.crew_file = crew_file
        self.flight_file = flight_file
        self.crews = None
        self.flights = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载机组和航班数据
        
        Returns:
            (crew_df, flight_df): 机组和航班数据框
        """
        # 加载机组数据
        self.crews = pd.read_csv(self.crew_file)
        
        # 加载航班数据
        self.flights = pd.read_csv(self.flight_file)
        
        # 预处理数据
        self._preprocess_crew_data()
        self._preprocess_flight_data()
        
        return self.crews, self.flights
    
    def _preprocess_crew_data(self):
        """预处理机组数据"""
        # 处理空值，将空字符串转换为False
        if 'Captain' in self.crews.columns:
            self.crews['Captain'] = self.crews['Captain'].fillna('').apply(lambda x: x == 'Y')
        if 'FirstOfficer' in self.crews.columns:
            self.crews['FirstOfficer'] = self.crews['FirstOfficer'].fillna('').apply(lambda x: x == 'Y')
        if 'Deadhead' in self.crews.columns:
            self.crews['Deadhead'] = self.crews['Deadhead'].fillna('').apply(lambda x: x == 'Y')
        
        # 添加派生字段
        self.crews['CanBeCaptain'] = self.crews.get('Captain', False)
        self.crews['CanBeFirstOfficer'] = self.crews.get('FirstOfficer', False)
        
    def _preprocess_flight_data(self):
        """预处理航班数据"""
        # 合并日期和时间，转换为datetime对象
        self.flights['DepartureDateTime'] = pd.to_datetime(
            self.flights['DptrDate'] + ' ' + self.flights['DptrTime'],
            format='%m/%d/%Y %H:%M'
        )
        self.flights['ArrivalDateTime'] = pd.to_datetime(
            self.flights['ArrvDate'] + ' ' + self.flights['ArrvTime'],
            format='%m/%d/%Y %H:%M'
        )
        
        # 计算飞行时间（分钟）
        self.flights['FlightDuration'] = (
            self.flights['ArrivalDateTime'] - self.flights['DepartureDateTime']
        ).dt.total_seconds() / 60
        
        # 提取日期用于分组
        self.flights['DepartureDate'] = self.flights['DepartureDateTime'].dt.date
        self.flights['ArrivalDate'] = self.flights['ArrivalDateTime'].dt.date
        
        # 解析机组配置需求 (C1F1 -> 1 Captain + 1 First Officer)
        self.flights['CaptainRequired'] = self.flights['Comp'].str.extract(r'C(\d+)')[0].astype(int)
        self.flights['FirstOfficerRequired'] = self.flights['Comp'].str.extract(r'F(\d+)')[0].astype(int)
        self.flights['TotalCrewRequired'] = (
            self.flights['CaptainRequired'] + self.flights['FirstOfficerRequired']
        )
        
        # 按时间排序
        self.flights = self.flights.sort_values('DepartureDateTime').reset_index(drop=True)
        
    def get_flight_compatibility(self, flight1_idx: int, flight2_idx: int, 
                                 min_connection_time: int) -> bool:
        """
        检查两个航班是否可以连接（用于构建配对）
        
        Args:
            flight1_idx: 第一个航班索引
            flight2_idx: 第二个航班索引
            min_connection_time: 最小连接时间（分钟）
            
        Returns:
            bool: 是否可以连接
        """
        if self.flights is None:
            return False
            
        f1 = self.flights.iloc[flight1_idx]
        f2 = self.flights.iloc[flight2_idx]
        
        # 检查地点匹配：第一个航班的到达站 = 第二个航班的出发站
        if f1['ArrvStn'] != f2['DptrStn']:
            return False
        
        # 检查时间间隔
        time_diff = (f2['DepartureDateTime'] - f1['ArrivalDateTime']).total_seconds() / 60
        
        return time_diff >= min_connection_time
    
    def get_unique_dates(self) -> List[datetime.date]:
        """获取所有唯一的日期"""
        if self.flights is None:
            return []
        return sorted(self.flights['DepartureDate'].unique())
    
    def get_unique_stations(self) -> List[str]:
        """获取所有唯一的机场"""
        if self.flights is None:
            return []
        stations = set(self.flights['DptrStn'].unique()) | set(self.flights['ArrvStn'].unique())
        return sorted(list(stations))
    
    def get_flights_by_date(self, date: datetime.date) -> pd.DataFrame:
        """获取特定日期的所有航班"""
        if self.flights is None:
            return pd.DataFrame()
        return self.flights[self.flights['DepartureDate'] == date]


def load_crew_scheduling_data(crew_file: str = None, flight_file: str = None):
    """
    便捷函数：加载机组排班数据
    
    Args:
        crew_file: 机组数据文件路径（可选）
        flight_file: 航班数据文件路径（可选）
        
    Returns:
        DataLoader对象
    """
    from crew_scheduling.config import CREW_DATA_FILE, FLIGHT_DATA_FILE
    
    if crew_file is None:
        crew_file = CREW_DATA_FILE
    if flight_file is None:
        flight_file = FLIGHT_DATA_FILE
        
    loader = DataLoader(crew_file, flight_file)
    loader.load_data()
    return loader
