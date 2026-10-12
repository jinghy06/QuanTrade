# A股量化信号系统 - 策略开发指南

## 快速开始（3步写一个策略）

### Step 1: 复制模板

```bash
cp strategies/examples/my_strategy_template.py strategies/custom/my_first_strategy.py
```

### Step 2: 填写策略逻辑

```python
from strategies.base import BaseStrategy, SignalResult

class DualMAStrategy(BaseStrategy):
    name = "dual_ma"                    # 英文标识
    display_name = "双均线策略"          # 中文名
    description = "5日/20日均线金叉买入，死叉卖出"
    category = "technical"
    author = "你的名字"

    required_columns = ["close"]        # 需要的数据列
    min_bars = 30                       # 最少K线数

    def __init__(self, fast=5, slow=20):
        super().__init__(fast=fast, slow=slow)
        self.fast = fast
        self.slow = slow

    def generate_signal(self, df):
        if not self.validate_data(df):
            return SignalResult(self.name, triggered=False)

        close = df["close"]
        ma_fast = close.rolling(self.fast).mean()
        ma_slow = close.rolling(self.slow).mean()

        # 金叉判断
        if ma_fast.iloc[-2] < ma_slow.iloc[-2] and ma_fast.iloc[-1] >= ma_slow.iloc[-1]:
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="buy",
                confidence=0.7,
                price=round(close.iloc[-1], 2),
                rationale=f"{self.fast}日均线上穿{self.slow}日均线形成金叉"
            )

        return SignalResult(self.name, triggered=False)
```

### Step 3: 运行

```python
from strategies.loader import StrategyLoader
from strategies.registry import StrategyRegistry

# 方式1: 自动加载
loader = StrategyLoader()
loader.load_all(custom_dir="strategies/custom")
registry = loader.get_registry()

# 方式2: 手动注册
from strategies.custom.my_first_strategy import DualMAStrategy
registry = StrategyRegistry()
registry.register(DualMAStrategy(fast=5, slow=20))

# 运行
results = registry.run_all(df_kline, symbol="000001.SZ")

# 查看信号
for name, signal in results.items():
    if signal:
        print(f"{name}: {signal.action} @ {signal.price}")

# 多策略投票聚合
combined = registry.aggregate_voting(results)
print(f"综合信号: {combined.action} 置信度:{combined.confidence}")
```

---

## 策略基类 API

### BaseStrategy 必须实现

| 属性/方法 | 类型 | 说明 |
|-----------|------|------|
| `name` | str | 英文标识，必须唯一 |
| `display_name` | str | 中文显示名 |
| `description` | str | 策略描述 |
| `category` | str | 分类: technical/fundamental/ml/combined |
| `required_columns` | List[str] | 最低数据列需求 |
| `min_bars` | int | 最少K线数 |
| `__init__(self, **kwargs)` | method | 初始化参数 |
| `generate_signal(self, df)` | method | **核心方法**，返回 SignalResult |

### BaseStrategy 可选覆盖

| 方法 | 说明 |
|------|------|
| `_validate_params()` | 参数校验 |
| `validate_data(df)` | 数据校验（通常无需覆盖） |
| `get_info()` | 获取策略信息字典 |

### SignalResult 字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `strategy_name` | str | 是 | 策略名 |
| `triggered` | bool | 是 | 是否触发 |
| `action` | str | 否 | buy/sell/hold/light_buy/light_sell |
| `confidence` | float | 否 | 0-1 置信度 |
| `price` | float | 否 | 当前价 |
| `target_price` | float | 否 | 目标价 |
| `stop_loss` | float | 否 | 止损价 |
| `position_pct` | float | 否 | 建议仓位 0-1 |
| `rationale` | str | 否 | 决策理由 |
| `risk_factors` | List[str] | 否 | 风险因素 |
| `metadata` | Dict | 否 | 自定义数据 |

---

## 内置策略清单

| 策略类 | 名称 | 说明 | 关键参数 |
|--------|------|------|---------|
| MACrossStrategy | 均线交叉 | 金叉买入/死叉卖出 | fast, slow |
| RSIStrategy | RSI超买超卖 | 超卖区买入/超买区卖出 | period, overbought, oversold |
| MACDStrategy | MACD金叉死叉 | DIF/DEA交叉+零轴判断 | fast, slow, signal |
| BreakoutStrategy | 突破策略 | 突破N日高/低点 | period, volume_confirm |
| MeanReversionStrategy | 均值回归 | 布林带偏离回归 | period, dev |
| MLHybridStrategy | ML混合 | LightGBM+技术面双重确认 | ml_trainer, min_ml_prob |

### 一键注册所有内置策略

```python
from strategies.loader import StrategyLoader

loader = StrategyLoader()
loader.load_built_in(
    ma_fast=5, ma_slow=20,      # 均线参数
    rsi_period=14,              # RSI参数
    breakout_period=20,         # 突破周期
)
```

---

## 策略注册中心 API

### 注册管理

```python
registry = StrategyRegistry()

# 注册
registry.register(MyStrategy(param1=10))

# 注销
registry.unregister("my_strategy")

# 查询
registry.has("my_strategy")        # bool
registry.get("my_strategy")        # BaseStrategy or None
registry.list_strategies()         # List[Dict] 策略信息列表
len(registry)                      # 策略数量
```

### 批量运行

```python
# 运行所有策略
results = registry.run_all(df_kline, symbol="000001.SZ")
# 返回: {策略名: SignalResult, ...}

# 投票聚合
combined = registry.aggregate_voting(results, method="confidence_weighted")
# method: "majority" | "confidence_weighted" | "unanimous"

# 冲突检测
conflicts = registry.check_conflicts(results)
```

---

## 策略开发最佳实践

### 1. 参数设计

```python
def __init__(self, period: int = 20, threshold: float = 0.5):
    # 所有参数都通过 kwargs 传递给父类保存
    super().__init__(period=period, threshold=threshold)
    self.period = period
    self.threshold = threshold
```

### 2. 防御性编程

```python
def generate_signal(self, df):
    # 必须: 数据校验
    if not self.validate_data(df):
        return SignalResult(self.name, triggered=False)

    # 建议: 计算前检查数据长度
    if len(df) < self.period + 5:
        return SignalResult(self.name, triggered=False, rationale="数据不足")

    # 建议: 指标计算保护
    indicator = some_value / (denominator + 1e-10)  # 防止除零
```

### 3. 元信息丰富化

```python
return SignalResult(
    strategy_name=self.name,
    triggered=True,
    action="buy",
    confidence=0.75,
    price=round(latest_close, 2),
    target_price=round(latest_close * 1.05, 2),  # 给LLM参考
    stop_loss=round(latest_close * 0.93, 2),      # 风控
    position_pct=0.05,                            # 仓位建议
    rationale="金叉+放量确认",                      # 人可读
    risk_factors=["假突破", "大盘风险"],             # 风险提示
    metadata={"rsi": 45, "macd": "golden"},        # 原始数据供调试
)
```

### 4. 与ML模型结合

```python
from strategies.built_in import MLHybridStrategy

# 需要已加载的MLTrainer
ml_strategy = MLHybridStrategy(
    ml_trainer=trainer,
    min_ml_prob=0.55,
    require_macd_align=True,
)
registry.register(ml_strategy)
```

---

## 策略文件组织

```
strategies/
├── __init__.py              # 包初始化
├── base.py                  # 基类 + SignalResult
├── registry.py              # 注册中心
├── loader.py                # 动态加载器
├── built_in.py              # 内置策略
├── examples/                # 示例模板
│   ├── __init__.py
│   └── my_strategy_template.py
├── custom/                  # 你的自定义策略放这里
│   ├── __init__.py
│   └── your_strategy.py
└── README.md                # 本文件
```

---

## 调试技巧

```python
# 单策略测试
strategy = MACrossStrategy(fast=5, slow=20)
print(strategy.get_info())

signal = strategy.generate_signal(df)
print(signal.to_dict())

# 查看触发历史
for record in registry._history[-5:]:
    print(f"{record['timestamp']} {record['symbol']}: {record['n_triggered']}/{record['n_strategies']}")
```
