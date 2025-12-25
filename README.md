# TA-Lib 技术指标计算 API 服务

一个基于 FastAPI 和 TA-Lib 的币安期货技术指标计算服务，支持并发处理、图片生成和合约交易功能。

## 功能特性

### 技术指标计算
- **趋势指标**: SMA、EMA、MACD
- **动量指标**: RSI、Stochastic RSI、ADX
- **波动性指标**: 布林带 (Bollinger Bands)、ATR
- **成交量指标**: OBV、VWAP、成交量移动平均
- **其他指标**: 斐波那契回调、蜡烛图形态识别（60+种形态）、海龟交易法则
- **市场数据**: 持仓量 (OI)、资金费率

### 图片生成
- 文本转图片功能
- 支持自定义字体、颜色、尺寸
- 支持多种图片格式（PNG、JPEG、WebP、BMP）
- 自动换行和文本布局

### 合约交易
- **账户信息查询**: 获取币安合约账户详情（固定20x杠杆）
- **账户信息 Markdown 格式**: 获取账户信息的 Markdown 格式字符串（供 Agent 使用）
- **交易信号处理**: 接收和处理交易信号（开仓、平仓、调整止盈止损等）
- **持仓管理**: 实时查询当前持仓信息，包括盈亏、杠杆、保证金等
- **精度控制**: 自动处理交易对的数量精度，避免 Invalid quantity 错误
- **Session 复用**: 优化 API 请求性能，减少 TCP 握手开销

### 技术特性
- 异步并发处理，支持多线程
- 自动过滤未完成的K线数据
- 智能数据合并（OI和资金费率）
- 完整的错误处理和重试机制
- RESTful API 设计
- 自动生成 API 文档

## 技术栈

- **Web框架**: FastAPI 0.104.0+
- **技术指标库**: TA-Lib 0.6.4
- **数据处理**: Pandas 2.0.0+, NumPy 1.24.0+
- **异步HTTP**: aiohttp 3.9.0+
- **图片处理**: Pillow 10.1.0+
- **数据验证**: Pydantic 2.0.0+
- **服务器**: Uvicorn

## 项目结构

```
TA-Lib/
├── app/
│   ├── __init__.py
│   ├── app.py                 # 主应用入口
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # 数据模型
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── indicators.py     # 技术指标路由
│   │   ├── images.py         # 图片生成路由
│   │   └── trading.py        # 交易相关路由
│   └── services/
│       ├── __init__.py
│       ├── indicator_service.py  # 技术指标计算服务
│       ├── image_service.py      # 图片生成服务
│       └── trading_service.py   # 交易服务
├── config/
│   └── config.json           # 配置文件
├── Dockerfile                # Docker 构建文件
├── docker-compose.yml        # Docker Compose 配置
├── requirements.txt          # Python 依赖
├── run.py                    # 启动脚本
├── deploy.sh                 # 部署脚本
└── 阿里巴巴普惠体.otf        # 中文字体文件
```

## 快速开始

### 环境要求

- Python 3.11+
- Docker 和 Docker Compose（可选）

### 本地开发

1. **克隆项目**
```bash
git clone <repository-url>
cd TA-Lib
```

2. **安装依赖**

**注意**: TA-Lib 需要先安装系统库，然后才能安装 Python 包。

**Linux/macOS:**
```bash
# 安装 TA-Lib 系统库
wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
tar -xzf ta-lib-0.6.4-src.tar.gz
cd ta-lib-0.6.4
./configure --prefix=/usr
make
sudo make install
cd ..

# 安装 Python 依赖
pip install -r requirements.txt
```

**Windows:**
```bash
# 下载预编译的 TA-Lib wheel 文件
# 或使用 conda: conda install -c conda-forge ta-lib
pip install -r requirements.txt
```

3. **配置**

编辑 `config/config.json`:
```json
{
  "binance_api_url": "https://fapi.binance.com",
  "thread_pool_size": 10,
  "request_timeout": 30,
  "retry_attempts": 3,
  "retry_delay": 1
}
```

**环境变量配置**（交易功能需要）:

创建 `.env` 文件在项目根目录（与 `run.py` 同级）:
```bash
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

或者使用系统环境变量：

Linux/macOS:
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

Windows PowerShell:
```powershell
$env:BINANCE_API_KEY="your_api_key"
$env:BINANCE_API_SECRET="your_api_secret"
```

**注意**: 推荐使用 `.env` 文件方式，系统会自动加载。需要安装 `python-dotenv` 库（已包含在 requirements.txt 中）。

4. **启动服务**
```bash
python run.py
```

服务将在 `http://localhost:8000` 启动。

### Docker 部署

1. **使用 Docker Compose（推荐）**
```bash
docker-compose up --build -d
```

2. **使用部署脚本**
```bash
chmod +x deploy.sh
./deploy.sh
```

3. **查看日志**
```bash
docker-compose logs -f
```

## API 文档

启动服务后，访问以下地址查看 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API 端点

### 健康检查

**GET** `/health`

返回服务健康状态。

**响应示例:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "version": "1.0.0"
}
```

### 获取支持的时间间隔

**GET** `/intervals`

获取支持的时间间隔列表。

**响应示例:**
```json
{
  "intervals": ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
  "description": "支持的时间间隔列表"
}
```

### 计算技术指标

**POST** `/calculate`

计算指定交易对的技术指标。

**请求体:**
```json
{
  "symbol": "BTCUSDT",
  "interval": "15m",
  "config": {
    "limit": 500,
    "rsi": 14,
    "sma": 20,
    "ema": 20,
    "macd": [12, 26, 9],
    "bb": [20, 2],
    "adx": 14,
    "stoch_rsi": {
      "period": 14,
      "kSmooth": 3,
      "dSmooth": 3
    },
    "atr": {
      "period": 14
    },
    "obv": true,
    "vwap": {
      "period": "day"
    },
    "volume_ma": 20,
    "fib": {
      "period": 100
    },
    "patterns": true,
    "turtle": {
      "entryPeriod": 20,
      "exitPeriod": 10,
      "atrPeriod": 14
    }
  }
}
```

**响应示例:**
```json
{
  "indicators": {
    "t": [1704067200000, 1704068100000, ...],
    "c": [42000.5, 42050.3, ...],
    "v": [1234567, 2345678, ...],
    "rsi": [65.5, 66.2, ...],
    "sma": [41900.0, 41950.0, ...],
    "ema": [41950.0, 42000.0, ...],
    "bb_u": [42500.0, 42550.0, ...],
    "bb_m": [42000.0, 42050.0, ...],
    "bb_l": [41500.0, 41550.0, ...],
    "macd": [50.5, 52.3, ...],
    "macd_s": [48.0, 49.5, ...],
    "macd_h": [2.5, 2.8, ...],
    "k": [75.5, 76.2, ...],
    "d": [73.0, 74.5, ...],
    "adx": [25.5, 26.0, ...],
    "atr": [500.0, 510.0, ...],
    "oi": [1234567890, 1234567891, ...],
    "funding": [0.0001, 0.0001, ...],
    "obv": [123456789, 234567890, ...],
    "vwap": [42000.0, 42025.0, ...],
    "v_ma": [1500000, 1600000, ...],
    "ptn": [["CDLENGULFING:100"], null, ...],
    "t_up": [42500.0, 42600.0, ...],
    "t_dn": [41500.0, 41600.0, ...],
    "t_sig": [0, 1, ...]
  },
  "fibonacci": {
    "fib_0236": 41800.0,
    "fib_0382": 41600.0,
    "fib_0500": 41400.0,
    "fib_0618": 41200.0,
    "fib_0786": 40900.0
  }
}
```

**字段说明:**
- `t`: 时间戳（毫秒）
- `c`: 收盘价
- `v`: 成交量
- `rsi`: RSI 指标
- `sma`: 简单移动平均
- `ema`: 指数移动平均
- `bb_u/m/l`: 布林带上/中/下轨
- `macd/macd_s/macd_h`: MACD 线/信号线/柱状图
- `k/d`: 随机 RSI 的 K/D 值
- `adx`: ADX 指标
- `atr`: ATR 指标
- `oi`: 持仓量
- `funding`: 资金费率
- `obv`: 能量潮指标
- `vwap`: 成交量加权平均价
- `v_ma`: 成交量移动平均
- `ptn`: 蜡烛图形态列表
- `t_up/t_dn/t_sig`: 海龟交易法则的上/下通道和信号

### 生成图片

**POST** `/generate`

将文本转换为图片。

**请求体:**
```json
{
  "text": "这是要转换为图片的文本内容\n支持多行文本",
  "width": 800,
  "font_size": 22,
  "margin": 40,
  "text_color": "#2D3748",
  "bg_color": "#FFFFFF",
  "format": "png",
  "quality": 95
}
```

**参数说明:**
- `text`: 文本内容（必填），支持 `\n` 换行
- `width`: 图片宽度（100-2000像素，默认750）
- `font_size`: 字体大小（8-72像素，默认20）
- `margin`: 边距（0-200像素，默认20）
- `text_color`: 文字颜色（十六进制，默认#2D3748）
- `bg_color`: 背景颜色（十六进制，默认#FFFFFF）
- `format`: 图片格式（png/jpeg/webp/bmp，默认png）
- `quality`: 图片质量（1-100，默认98，仅对jpeg/webp有效）

**响应:** 返回图片文件流

### 获取合约账户信息

**GET** `/account`

获取币安合约账户详情，包括账户摘要、持仓信息和仓位范围（固定20x杠杆）。

**注意**: 需要配置 `BINANCE_API_KEY` 和 `BINANCE_API_SECRET` 环境变量。

**响应示例:**
```json
{
  "timestamp": 1704067200000,
  "single_coin_position": "山寨62500-175000 U(19x杠杆) | BTC/ETH 125000-237500 U(19x杠杆)",
  "account_summary": "净值12500.50 | 余额8500.25 (68.0%) | 盈亏+2.45% | 保证金32.0% | 持仓1个",
  "positions": [
    {
      "symbol": "BTCUSDT",
      "side": "LONG",
      "entry_price": 68123.45,
      "current_price": 68500.12,
      "pnl_percent": 2.35,
      "leverage": 20,
      "margin": 4000.0,
      "liquidation_price": 62000.0,
      "duration_minutes": 45,
      "stop_loss": 67500.0,
      "take_profit": 72000.0,
      "funding_rate": 0.01,
      "display_string": "BTCUSDT LONG | 入场价68123.4500 当前价68500.1200 | 盈亏+2.35% | 杠杆20x | 保证金4000 | 强平价62000.0000 | 持仓时长45分钟 | 止损67500.0000 止盈72000.0000 | 资金费率0.0100%"
    }
  ]
}
```

**字段说明:**
- `timestamp`: 时间戳（毫秒）
- `single_coin_position`: 单币仓位范围说明（根据账户净值动态计算）
- `account_summary`: 账户摘要（净值、余额、盈亏、保证金、持仓数量）
- `positions`: 当前持仓列表
  - `symbol`: 交易对符号
  - `side`: 方向（LONG/SHORT）
  - `entry_price`: 入场价格
  - `current_price`: 当前标记价格
  - `pnl_percent`: 盈亏百分比
  - `leverage`: 杠杆倍数
  - `margin`: 保证金
  - `liquidation_price`: 强平价格
  - `duration_minutes`: 持仓时长（分钟）
  - `stop_loss`: 止损价格（如有）
  - `take_profit`: 止盈价格（如有）
  - `funding_rate`: 资金费率（%）
  - `display_string`: 格式化显示字符串

### 获取账户信息 Markdown 格式

**GET** `/account/markdown`

获取币安合约账户信息的 Markdown 格式字符串，专为 Agent 使用设计。

**注意**: 需要配置 `BINANCE_API_KEY` 和 `BINANCE_API_SECRET` 环境变量。

**响应:** 返回 Markdown 格式的文本内容（`text/markdown` 媒体类型）

**响应示例:**
```markdown
账户详情：

时间: 1704067200000

单币仓位: 山寨62500-175000 U(19x杠杆) | BTC/ETH 125000-237500 U(19x杠杆)

账户: 净值12500.50 | 余额8500.25 (68.0%) | 盈亏+2.45% | 保证金32.0% | 持仓1个

## 当前持仓

1. BTCUSDT LONG | 入场价68123.4500 当前价68500.1200 | 盈亏+2.35% | 杠杆20x | 保证金4000 | 强平价62000.0000 | 持仓时长45分钟 | 止损67500.0000 止盈72000.0000 | 资金费率0.0100%
```

**格式说明:**
- 时间戳为毫秒格式
- 单币仓位范围根据账户净值动态计算
- 账户摘要包含净值、余额、盈亏、保证金和持仓数量
- 持仓列表包含所有持仓的详细信息

### 接收交易信号

**POST** `/signals`

接收并处理交易信号，支持开仓、平仓、调整止盈止损等操作。

**请求体:**
```json
{
  "signals": [
    {
      "symbol": "SOLUSDT",
      "action": "wait",
      "reasoning": "价格处于震荡区间，等待明确方向"
    },
    {
      "symbol": "BTCUSDT",
      "action": "open_long",
      "leverage": 20,
      "position_size_usd": 200,
      "stop_loss": 68000,
      "take_profit": 72000,
      "confidence": 90,
      "risk_usd": 150,
      "reasoning": "突破关键阻力位，看涨信号强烈"
    },
    {
      "symbol": "ETHUSDT",
      "action": "open_short",
      "leverage": 20,
      "position_size_usd": 200,
      "stop_loss": 3100,
      "take_profit": 2900,
      "confidence": 95,
      "risk_usd": 100,
      "reasoning": "接近阻力位，看跌信号"
    },
    {
      "symbol": "BTCUSDT",
      "action": "adjust_stops",
      "stop_loss": 69500,
      "take_profit": 75000,
      "reasoning": "价格上移，调整止损至盈亏平衡点"
    },
    {
      "symbol": "BTCUSDT",
      "action": "close_long",
      "reasoning": "达到目标价位，平仓获利"
    }
  ]
}
```

**响应示例:**
```json
{
  "processed_count": 5,
  "results": [
    {
      "symbol": "SOLUSDT",
      "action": "wait",
      "status": "skipped",
      "message": "观望: 价格处于震荡区间，等待明确方向"
    },
    {
      "symbol": "BTCUSDT",
      "action": "open_long",
      "status": "success",
      "message": "开仓成功",
      "order_id": 12345678
    },
    {
      "symbol": "ETHUSDT",
      "action": "open_short",
      "status": "success",
      "message": "开仓成功",
      "order_id": 12345679
    },
    {
      "symbol": "BTCUSDT",
      "action": "adjust_stops",
      "status": "success",
      "message": "止损止盈已更新"
    },
    {
      "symbol": "BTCUSDT",
      "action": "close_long",
      "status": "success",
      "message": "已平仓"
    }
  ]
}
```

**操作类型说明:**

| action | 说明 | 必填字段 | 说明 |
|--------|------|---------|------|
| `open_long` | 开多仓 | `position_size_usd`, `stop_loss`, `take_profit`, `confidence`, `risk_usd` | 信心度需≥90，杠杆固定20x |
| `open_short` | 开空仓 | 同上 | 同上 |
| `close_long` | 平多仓 | `reasoning`（可选） | 平掉所有多仓 |
| `close_short` | 平空仓 | `reasoning`（可选） | 平掉所有空仓 |
| `adjust_stops` | 调整止盈止损 | `stop_loss`, `take_profit` | 需要已有持仓 |
| `hold` | 持有 | `reasoning`（可选） | 保持当前持仓不变 |
| `wait` | 等待 | `reasoning`（可选） | 观望，不进行任何操作 |

**字段说明:**
- `symbol`: 交易对符号（如 BTCUSDT）
- `action`: 操作类型（见上表）
- `position_size_usd`: 仓位大小（USD，开仓时必填）
- `stop_loss`: 止损价格（开仓或调整时必填）
- `take_profit`: 止盈价格（开仓或调整时必填）
- `confidence`: 信心度 0-100（开仓时必填，需≥90）
- `risk_usd`: 风险金额（USD，开仓时必填）
- `reasoning`: 操作原因说明（可选，建议填写）

**注意事项:**
1. **杠杆固定**: 所有开仓操作使用固定 20x 杠杆，无需在信号中指定
2. **精度控制**: 系统会自动根据交易对的 `stepSize` 精度要求调整下单数量，避免 Invalid quantity 错误
3. **止盈止损**: 开仓操作会自动设置止盈止损单，使用 `MARK_PRICE` 工作类型
4. **调整止盈止损**: 会先撤销旧的止盈止损单，再挂新的
5. **平仓操作**: 会先撤销该交易对的所有挂单，再执行市价平仓，数量会自动按精度要求调整
6. **等待和持有**: `wait` 和 `hold` 操作不会执行任何交易，仅记录日志
7. **API 权限**: 所有操作都需要有效的API密钥和合约交易权限

## 配置说明

### config.json

```json
{
  "binance_api_url": "https://fapi.binance.com",  // 币安API地址
  "thread_pool_size": 10,                         // 线程池大小
  "request_timeout": 30,                          // 请求超时时间（秒）
  "retry_attempts": 3,                            // 重试次数
  "retry_delay": 1                                // 重试延迟（秒）
}
```

## 支持的时间间隔

- `1m` - 1分钟
- `5m` - 5分钟
- `15m` - 15分钟
- `30m` - 30分钟
- `1h` - 1小时
- `2h` - 2小时
- `4h` - 4小时
- `1d` - 1天

## 技术指标配置说明

### 基础指标

- `rsi`: RSI 周期（默认14）
- `sma`: SMA 周期（默认20）
- `ema`: EMA 周期（默认20）
- `adx`: ADX 周期（默认14）

### MACD

```json
"macd": [12, 26, 9]  // [快线周期, 慢线周期, 信号线周期]
```

### 布林带

```json
"bb": [20, 2]  // [周期, 标准差倍数]
```

### 随机 RSI

```json
"stoch_rsi": {
  "period": 14,      // RSI 周期
  "kSmooth": 3,      // K 值平滑周期
  "dSmooth": 3       // D 值平滑周期
}
```

### ATR

```json
"atr": {
  "period": 14  // ATR 周期
}
```

### VWAP

```json
"vwap": {
  "period": "day"  // "day" 表示按日重置，null 表示累计
}
```

### 斐波那契回调

```json
"fib": {
  "period": 100  // 计算最近N根K线的斐波那契回调
}
```

### 海龟交易法则

```json
"turtle": {
  "entryPeriod": 20,   // 入场通道周期
  "exitPeriod": 10,     // 出场通道周期
  "atrPeriod": 14       // ATR 周期
}
```

## 注意事项

1. **TA-Lib 安装**: TA-Lib 需要先安装系统库，然后才能安装 Python 包。Docker 镜像已包含编译好的 TA-Lib。

2. **数据过滤**: 服务会自动过滤未完成的K线数据，确保计算结果的准确性。

3. **并发处理**: 服务使用线程池处理请求，默认线程数为10，可在配置文件中调整。

4. **数据精度**: 
   - 时间戳和成交量数据会转换为整数以节省 Token
   - 价格和指标数据保留8位小数精度

5. **持仓量数据**: 
   - 1分钟周期会自动降级使用5分钟OI数据
   - 不同周期使用不同的时间容差进行数据合并

6. **字体文件**: 图片生成功能需要 `阿里巴巴普惠体.otf` 字体文件，如果缺失会使用系统默认字体。

7. **交易功能配置**: 
   - 交易功能需要配置 `BINANCE_API_KEY` 和 `BINANCE_API_SECRET` 环境变量（推荐使用 `.env` 文件）
   - 确保API密钥具有合约交易权限
   - 杠杆固定为20x（实际交易），账户信息显示为19x（安全杠杆）
   - 仓位范围根据账户净值动态计算：
     - BTC/ETH: 最小 = 净值 × 10，最大 = 净值 × 19
     - 山寨币: 最小 = 净值 × 5，最大 = 净值 × 19
   - 系统使用 Session 复用优化，减少 TCP 握手开销
   - 自动处理交易对精度，避免 Invalid quantity 错误

8. **交易信号处理**:
   - 开仓操作要求信心度≥90，否则会被拒绝
   - `wait` 和 `hold` 操作不会执行任何交易，仅记录日志
   - 开仓时会自动设置杠杆（固定20x）并挂止盈止损单（使用 MARK_PRICE 工作类型）
   - 调整止盈止损会先撤销旧单再挂新单
   - 平仓操作会先撤销所有挂单再执行市价平仓，数量会自动按精度要求调整
   - 所有下单数量会根据交易对的 `stepSize` 自动调整精度

9. **API安全**:
   - 所有交易API请求都使用HMAC-SHA256签名
   - 建议使用只读权限的API密钥进行账户查询
   - 生产环境建议使用IP白名单限制API访问

## 开发

### 代码结构

- `app/app.py`: FastAPI 应用主入口，配置中间件和路由
- `app/routers/`: API 路由定义
- `app/services/`: 业务逻辑服务
- `app/models/`: 数据模型定义

### 日志

服务使用 Python 标准 logging 模块，日志级别为 INFO，输出到控制台。

## 许可证

[根据实际情况填写]

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

[根据实际情况填写]
