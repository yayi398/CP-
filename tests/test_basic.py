# -*- coding: utf-8 -*-
"""
基本测试
Basic tests for crew scheduling system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crew_scheduling.data_loader import DataLoader
from crew_scheduling import config


def test_data_loading():
    """测试数据加载"""
    print("测试数据加载...")
    
    loader = DataLoader(config.CREW_DATA_FILE, config.FLIGHT_DATA_FILE)
    crews, flights = loader.load_data()
    
    assert len(crews) > 0, "机组数据为空"
    assert len(flights) > 0, "航班数据为空"
    
    print(f"✓ 成功加载 {len(crews)} 名机组成员")
    print(f"✓ 成功加载 {len(flights)} 个航班")
    
    # 检查必要的列
    assert 'EmpNo' in crews.columns
    assert 'Base' in crews.columns
    assert 'DutyCostPerHour' in crews.columns
    
    assert 'FltNum' in flights.columns
    assert 'DepartureDateTime' in flights.columns
    assert 'ArrivalDateTime' in flights.columns
    assert 'FlightDuration' in flights.columns
    
    print("✓ 数据结构验证通过")
    
    return loader


def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    from crew_scheduling.model import create_simple_model
    
    loader = DataLoader(config.CREW_DATA_FILE, config.FLIGHT_DATA_FILE)
    crews, flights = loader.load_data()
    
    cfg = {
        'MIN_CONNECTION_TIME': config.MIN_CONNECTION_TIME,
        'MAX_DUTY_PERIOD': config.MAX_DUTY_PERIOD,
    }
    
    prob, x = create_simple_model(flights, crews, cfg)
    
    assert prob is not None, "模型创建失败"
    assert len(x) > 0, "决策变量为空"
    
    print(f"✓ 成功创建模型，包含 {len(x)} 个决策变量")
    
    return prob, x


def test_pairing_generation():
    """测试配对生成"""
    print("\n测试配对生成...")
    
    from crew_scheduling.utils.pairing_generator import generate_simple_pairings
    
    loader = DataLoader(config.CREW_DATA_FILE, config.FLIGHT_DATA_FILE)
    crews, flights = loader.load_data()
    
    base = crews.iloc[0]['Base']
    
    pairings = generate_simple_pairings(
        flights,
        base,
        config.MIN_CONNECTION_TIME,
        config.MAX_PAIRING_DAYS
    )
    
    print(f"✓ 成功生成 {len(pairings)} 个配对")
    
    if pairings:
        print(f"  示例配对: {pairings[0]}")
    
    return pairings


if __name__ == '__main__':
    print("=" * 60)
    print("机组排班系统基本测试")
    print("=" * 60)
    
    try:
        loader = test_data_loading()
        prob, x = test_model_creation()
        pairings = test_pairing_generation()
        
        print("\n" + "=" * 60)
        print("所有测试通过 ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
