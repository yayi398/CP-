# -*- coding: utf-8 -*-
"""
配置参数
Configuration parameters for crew pairing and rostering optimization
基于论文 Saemi et al., 2021 和问题描述
"""

# 时间约束参数 (分钟)
MIN_CONNECTION_TIME = 40  # 最小过站时间 (分钟)
MAX_DAILY_FLIGHT_TIME = 480  # 最大日飞行时间 (8小时 = 480分钟)
MAX_DUTY_PERIOD = 720  # 最大执勤时间 (12小时 = 720分钟)
MIN_REST_TIME = 600  # 最小休息时间 (10小时 = 600分钟)
MAX_PAIRING_DAYS = 4  # 最大配对持续天数

# 从README获取的额外参数
MAX_BLOCK_TIME = 600  # 一次执勤飞行时长最多不超过 (分钟)
MIN_REST_BETWEEN_DUTIES = 660  # 相邻执勤之间的休息时间不少于 (分钟)
MAX_DEADHEAD = 5  # 每趟航班最多乘机人数
MAX_TAFB = 14400  # 排班周期单个机组人员任务环总时长不超过 (分钟，即10天)
MAX_SUCCESSIVE_ON = 4  # 连续执勤天数不超过 (天)
MIN_VACATION_DAYS = 2  # 相邻两个任务环之间至少休息天数

# 机组配置
CREW_PER_FLIGHT = 2  # 每个航班需要的机组人数 (基于C1F1配置)

# 数据文件路径
CREW_DATA_FILE = '机组排班Data A-Crew.csv'
FLIGHT_DATA_FILE = '机组排班Data A-Flight.csv'

# 求解器参数
# MILP求解器参数
MILP_TIME_LIMIT = 300  # MILP求解时间限制(秒)
MILP_GAP = 0.05  # 可接受的优化间隙

# 遗传算法参数
GA_POPULATION_SIZE = 100  # 种群大小
GA_GENERATIONS = 100  # 迭代代数
GA_CROSSOVER_RATE = 0.8  # 交叉率
GA_MUTATION_RATE = 0.2  # 变异率

# 粒子群优化参数
PSO_SWARM_SIZE = 50  # 粒子群大小
PSO_ITERATIONS = 100  # 迭代次数
PSO_W = 0.7  # 惯性权重
PSO_C1 = 1.5  # 个体学习因子
PSO_C2 = 1.5  # 社会学习因子

# 成本权重
DUTY_COST_WEIGHT = 1.0  # 执勤成本权重
PAIRING_COST_WEIGHT = 1.0  # 配对成本权重
DEADHEAD_COST_WEIGHT = 1.5  # 空乘成本权重(通常更高)

# 输出文件
OUTPUT_UNCOVERED_FLIGHTS = 'UncoveredFlights.csv'
OUTPUT_CREW_ROSTERS = 'CrewRosters.csv'
OUTPUT_RESULTS_DIR = 'results'
