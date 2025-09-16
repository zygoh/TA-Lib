# 币安技术指标计算API服务

基于TA-Lib的高性能币安期货技术指标计算服务，支持并发处理。

## 文件结构

```
├── app.py          # Web API服务（FastAPI）
├── indicators.py   # 币安API调用 + 技术指标计算（使用TA-Lib）
├── config.json     # 配置文件
├── requirements.txt # Python依赖
└── README.md       # 说明文档
```

### 职责分工

- **`app.py`** - 纯Web服务，只处理HTTP请求和响应
- **`indicators.py`** - 核心业务逻辑，包含币安API调用和技术指标计算
- **`config.json`** - 配置文件，存储API地址和系统参数

## 功能特性

- 🚀 **高性能**: 使用TA-Lib库，计算速度快、精度高
- 🔄 **并发支持**: 异步处理，支持批量计算
- 📊 **丰富指标**: 支持RSI、MACD、布林带、ADX等主流指标
- 🌐 **RESTful API**: 标准HTTP接口，易于集成
- 📝 **完整日志**: 生产级日志记录
- 🔍 **健康检查**: 内置健康检查机制

## 支持的技术指标

- **RSI** - 相对强弱指数
- **SMA** - 简单移动平均线  
- **EMA** - 指数移动平均线
- **Bollinger Bands** - 布林带
- **MACD** - 移动平均收敛散度
- **Stochastic RSI** - 随机RSI
- **ADX** - 平均趋向指数

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行服务

```bash
python app.py
```

### 访问API文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API接口

### 1. 健康检查

```bash
GET /health
```

### 2. 获取支持的时间间隔

```bash
GET /intervals
```

### 3. 计算单个技术指标

```bash
POST /calculate
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "interval": "15m",
  "config": {
    "limit": 150,
    "rsi": 12,
    "macd": [12, 26, 9],
    "bb": [20, 2],
    "sma": 20,
    "ema": 12,
    "adx": 10,
    "stoch_rsi": {
      "period": 12,
      "kSmooth": 3,
      "dSmooth": 3
    }
  }
}
```

### 4. 批量计算技术指标

```bash
POST /calculate/batch
Content-Type: application/json

{
  "requests": [
    {
      "symbol": "BTCUSDT",
      "interval": "15m",
      "config": { ... }
    },
    {
      "symbol": "ETHUSDT", 
      "interval": "1h",
      "config": { ... }
    }
  ]
}
```

## 配置参数说明

### 基础参数
- `limit`: 获取K线数据条数
- `symbol`: 交易对符号（如BTCUSDT）
- `interval`: 时间间隔（15m, 30m, 1h, 2h, 4h, 1d）

### 技术指标参数
- `rsi`: RSI周期
- `sma`: SMA周期
- `ema`: EMA周期
- `adx`: ADX周期
- `macd`: [快线周期, 慢线周期, 信号线周期]
- `bb`: [周期, 标准差倍数]
- `stoch_rsi`: {period: 周期, kSmooth: K平滑, dSmooth: D平滑}

## 配置文件

编辑 `config.json` 文件来配置系统参数：

```json
{
  "binance_api_url": "http://43.128.89.167",
  "thread_pool_size": 10,
  "request_timeout": 30
}
```

### 配置参数说明

- **`binance_api_url`**: 币安API完整地址（包含代理）
- **`thread_pool_size`**: 线程池大小（并发请求数）
- **`request_timeout`**: 请求超时时间（秒）

## 使用示例

```python
import requests

# 计算BTCUSDT 15分钟的技术指标
response = requests.post('http://localhost:8000/calculate', json={
    "symbol": "BTCUSDT",
    "interval": "15m",
    "config": {
        "limit": 150,
        "rsi": 12,
        "macd": [12, 26, 9],
        "bb": [20, 2],
        "sma": 20,
        "ema": 12,
        "adx": 10,
        "stoch_rsi": {
            "period": 12,
            "kSmooth": 3,
            "dSmooth": 3
        }
    }
})

result = response.json()
print(f"最新RSI: {result['summary']['rsi']:.2f}")
print(f"最新价格: {result['price_data']['close'][-1]}")
```

## 许可证

MIT License