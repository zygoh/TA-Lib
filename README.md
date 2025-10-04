# TA-Lib API 项目

币安技术指标计算API服务，支持并发计算和图片生成功能。

## 项目结构

```
project/
├── app/                        # 主应用包
│   ├── __init__.py
│   ├── main.py                # 主应用入口
│   ├── routers/               # 路由模块
│   │   ├── __init__.py
│   │   ├── indicators.py      # 技术指标路由
│   │   └── images.py          # 图片生成路由
│   ├── services/              # 业务服务层
│   │   ├── __init__.py
│   │   ├── indicator_service.py # 技术指标计算服务
│   │   └── image_service.py     # 图片生成服务
│   └── models/                # 数据模型
│       ├── __init__.py
│       └── schemas.py         # Pydantic模型定义
├── config/                    # 配置文件
│   └── config.json           # 应用配置
├── main.py                   # 应用启动脚本
├── requirements.txt          # Python依赖
├── Dockerfile               # Docker配置
├── docker-compose.yml       # Docker Compose配置
├── deploy.sh               # 部署脚本
└── README.md              # 项目说明
```

## 主要功能

### 1. 技术指标计算
- **端点**: `POST /calculate`
- **功能**: 计算各种技术指标（SMA, EMA, RSI, MACD等）
- **支持**: 并发计算，多种时间间隔

### 2. 图片生成
- **端点**: `POST /generate`
- **功能**: 将文本转换为图片
- **支持**: 多种格式（PNG, JPEG, WebP, BMP），自定义样式

### 3. 系统信息
- **端点**: `GET /health` - 健康检查
- **端点**: `GET /intervals` - 获取支持的时间间隔

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动服务
```bash
python main.py
```

### 3. 访问API文档
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 配置说明

配置文件位于 `config/config.json`：

```json
{
    "binance_api_url": "https://fapi.binance.com",
    "thread_pool_size": 10,
    "request_timeout": 30,
    "retry_attempts": 3,
    "retry_delay": 1
}
```

## 部署

### 生产环境部署
```bash
# 使用部署脚本（推荐）
bash deploy.sh

# 或手动部署
docker-compose up --build -d
```

### 开发环境部署
```bash
# 使用开发配置（支持热重载）
docker-compose -f docker-compose.dev.yml up --build -d
```

### 本地开发
```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python run.py

# 或者直接使用uvicorn
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

## API使用示例

### 计算技术指标
```bash
curl -X POST "http://localhost:8000/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "interval": "1h",
    "config": {
      "sma": {"period": 20},
      "rsi": {"period": 14}
    }
  }'
```

### 生成图片
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello World",
    "width": 800,
    "font_size": 24
  }' \
  --output image.png
```

## 技术栈

- **框架**: FastAPI
- **技术指标**: TA-Lib
- **数据处理**: Pandas, NumPy
- **图片生成**: Pillow
- **异步处理**: asyncio, aiohttp
- **部署**: Docker, uvicorn
