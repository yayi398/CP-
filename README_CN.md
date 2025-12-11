# 机组配对与排班优化系统

基于论文 "A new mathematical model to cover crew pairing and rostering problems simultaneously" (Saemi et al., 2021) 的实现。

## 项目简介

本项目实现了航空公司机组配对与排班的集成优化模型，提供三种求解方法：
1. **MILP精确求解** - 使用PuLP + CBC求解器
2. **遗传算法 (GA)** - 适用于大规模问题
3. **粒子群优化 (PSO)** - 论文表明效果最好

## 项目结构

```
CP-/
├── crew_scheduling/              # 主程序包
│   ├── __init__.py
│   ├── config.py                # 配置参数
│   ├── data_loader.py          # 数据加载和预处理
│   ├── model.py                # MILP数学模型定义
│   ├── main.py                 # 主程序入口
│   ├── solvers/                # 求解器模块
│   │   ├── milp_solver.py     # MILP精确求解
│   │   ├── ga_solver.py       # 遗传算法
│   │   └── pso_solver.py      # 粒子群优化
│   └── utils/                  # 工具模块
│       ├── pairing_generator.py  # 配对生成
│       └── visualization.py      # 结果可视化
├── tests/                       # 测试文件
│   └── test_basic.py
├── 机组排班Data A-Crew.csv      # 机组数据 (21名成员)
├── 机组排班Data A-Flight.csv    # 航班数据 (2021-08-11至08-25)
├── requirements.txt             # 依赖包
└── README.md                    # 本文件
```

## 数据文件说明

### 机组数据 (机组排班Data A-Crew.csv)
- **EmpNo**: 员工编号
- **Captain**: 是否有机长资格 (Y/空)
- **FirstOfficer**: 是否有副驾驶资格 (Y/空)
- **Deadhead**: 是否允许乘机摆渡 (Y/空)
- **Base**: 基地机场代码
- **DutyCostPerHour**: 执勤成本/小时 (元)
- **ParingCostPerHour**: 配对成本/小时 (元)

### 航班数据 (机组排班Data A-Flight.csv)
- **FltNum**: 航班号
- **DptrDate**: 出发日期
- **DptrTime**: 出发时间
- **DptrStn**: 出发机场
- **ArrvDate**: 到达日期
- **ArrvTime**: 到达时间
- **ArrvStn**: 到达机场
- **Comp**: 机组配置需求 (如C1F1表示1名机长+1名副驾驶)

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖包：
- pandas - 数据处理
- numpy - 数值计算
- pulp - MILP建模和求解
- matplotlib - 可视化

## 使用方法

### 1. 运行主程序

```bash
# 使用所有三种方法求解并比较
python -m crew_scheduling.main --method all --visualize

# 仅使用MILP方法
python -m crew_scheduling.main --method milp

# 仅使用遗传算法
python -m crew_scheduling.main --method ga

# 仅使用粒子群优化
python -m crew_scheduling.main --method pso

# 指定输出目录
python -m crew_scheduling.main --method all --output-dir my_results --visualize
```

### 2. 命令行参数

- `--method`: 求解方法 (milp/ga/pso/all)，默认为all
- `--crew-file`: 机组数据文件路径，默认为'机组排班Data A-Crew.csv'
- `--flight-file`: 航班数据文件路径，默认为'机组排班Data A-Flight.csv'
- `--output-dir`: 输出目录，默认为'results'
- `--visualize`: 生成可视化图表

### 3. 编程接口示例

```python
from crew_scheduling.data_loader import DataLoader
from crew_scheduling.solvers.milp_solver import MILPSolver
from crew_scheduling.solvers.ga_solver import GASolver
from crew_scheduling.solvers.pso_solver import PSOSolver
from crew_scheduling import config

# 加载数据
loader = DataLoader(
    '机组排班Data A-Crew.csv',
    '机组排班Data A-Flight.csv'
)
crews, flights = loader.load_data()

# 准备配置
cfg = {
    'MIN_CONNECTION_TIME': config.MIN_CONNECTION_TIME,
    'MAX_DUTY_PERIOD': config.MAX_DUTY_PERIOD,
    # ... 其他配置参数
}

# 使用MILP求解
milp_solver = MILPSolver(flights, crews, cfg)
result = milp_solver.solve(use_simple_model=True)

# 使用遗传算法求解
ga_solver = GASolver(flights, crews, cfg)
result = ga_solver.solve()

# 使用粒子群优化求解
pso_solver = PSOSolver(flights, crews, cfg)
result = pso_solver.solve()
```

## 数学模型

### 集合定义
- **F**: 航班集合
- **K**: 机组成员集合
- **P**: 配对(pairing)集合
- **D**: 日期集合

### 决策变量
- **x[k,p]**: 二元变量，机组k是否被分配到配对p
- **y[f,p]**: 二元变量，航班f是否包含在配对p中
- **z[f,k]**: 二元变量，机组k是否执行航班f

### 目标函数
最小化总成本 = 执勤成本 + 配对成本 + 空乘成本

### 主要约束
1. **航班覆盖约束**: 每个航班必须被恰好覆盖所需次数
2. **机组可用性约束**: 每个机组在同一时间只能执行一个任务
3. **配对连续性约束**: 配对内航班的时间和地点必须衔接
4. **最大飞行时间约束**: 每日最大飞行时间限制
5. **休息时间约束**: 两个执勤日之间的最小休息时间
6. **基地约束**: 配对必须从基地开始并返回基地

## 参数配置

主要参数在 `crew_scheduling/config.py` 中定义：

```python
MIN_CONNECTION_TIME = 40        # 最小过站时间 (分钟)
MAX_DAILY_FLIGHT_TIME = 480     # 最大日飞行时间 (8小时)
MAX_DUTY_PERIOD = 720           # 最大执勤时间 (12小时)
MIN_REST_TIME = 600             # 最小休息时间 (10小时)
MAX_PAIRING_DAYS = 4            # 最大配对持续天数
MAX_BLOCK_TIME = 600            # 一次执勤最大飞行时间 (分钟)
MIN_REST_BETWEEN_DUTIES = 660   # 相邻执勤间最小休息 (11小时)
MAX_TAFB = 14400               # 任务环最大时长 (10天)
MAX_SUCCESSIVE_ON = 4           # 最大连续执勤天数
MIN_VACATION_DAYS = 2           # 任务环间最小休息天数
```

## 输出文件

程序运行后会在输出目录生成以下文件：

1. **CrewRosters.csv** - 机组排班表
   - 包含每个机组的航班分配详情
   
2. **UncoveredFlights.csv** - 未覆盖的航班列表
   - 列出无法满足机组配置的航班
   
3. **可视化图表** (使用--visualize参数时)
   - gantt_chart.png - 甘特图展示机组排班
   - crew_utilization.png - 机组工作时长分布
   - cost_breakdown.png - 成本分解饼图

## 算法说明

### 1. MILP精确求解
- 使用PuLP建立混合整数线性规划模型
- 采用CBC开源求解器
- 适用于小到中等规模问题
- 可以找到最优解或近优解

### 2. 遗传算法 (GA)
- 种群大小: 100
- 迭代代数: 100
- 交叉率: 0.8
- 变异率: 0.2
- 适用于大规模问题
- 可以快速找到较好的可行解

### 3. 粒子群优化 (PSO)
- 粒子群大小: 50
- 迭代次数: 100
- 惯性权重: 0.7
- 学习因子: c1=1.5, c2=1.5
- 根据论文，效果最好
- 收敛速度快，解质量高

## 测试

运行基本测试：
```bash
python tests/test_basic.py
```

测试内容包括：
- 数据加载验证
- 模型创建验证
- 配对生成验证

## 性能优化建议

1. **小规模问题** (< 100航班): 使用MILP精确求解
2. **中等规模问题** (100-500航班): 使用PSO或GA
3. **大规模问题** (> 500航班): 使用GA，调整种群大小和代数

## 注意事项

1. 本实现为教学和研究目的，实际应用需要进一步优化
2. 大规模问题求解时间较长，建议先用小数据集测试
3. 可视化需要matplotlib支持中文字体
4. MILP求解器需要安装CBC (PuLP会自动下载)

## 参考文献

Saemi, S., Nourelfath, M., & Zaraatian, M. (2021). "A new mathematical model to cover crew pairing and rostering problems simultaneously". Journal of Engineering Research, 9(3B), 1-18.

## 许可证

本项目仅供学习和研究使用。

## 作者

Crew Scheduling Team

## 更新日志

### v1.0.0 (2024)
- 初始版本
- 实现MILP、GA、PSO三种求解方法
- 支持数据加载、模型构建、求解和可视化
- 提供命令行和编程两种接口
