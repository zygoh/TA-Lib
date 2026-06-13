# TA-Lib

本仓库是 **币安 U 本位合约数据与技术指标的 HTTP 服务**。单一 FastAPI 应用提供：技术指标计算、K 线图表、交易/账户代理、**`crypto-mcp`**（行情聚合、情绪/新闻、K 线出图、多平台分发）、**`maternal-mcp`**（Telegram 与公众号草稿）、以及 **`/images/clean`**（AI 生图元数据清洗）。**不再内置对 xAI Grok 的调用。**

> 注意：Python 依赖会安装官方技术分析库 **`TA-Lib`**（C 库绑定）。文档中「本服务」指本 FastAPI 应用；「ta-lib 库」指技术指标 C 库。

## 主要能力

| 模块 | 作用 |
|------|------|
| **健康检查** | `GET /`、`GET /health` |
| **技术指标** | 币安期货 K 线指标计算（`POST /calculate` 等） |
| **交易与账户** | 配置币安密钥后拉取账户、处理交易信号 |
| **图片** | 文本出图 `POST /generate`；**`POST /images/clean`** 清洗 AI 生图元数据（C2PA/EXIF，standard 4-pass → JPEG） |
| **crypto-mcp** | Agent 聚合：时间、涨幅榜、bundle、情绪/新闻（免费 RSS）、K 线生成与取图、**TG + X + Square 分发** |
| **maternal-mcp** | Telegram 图文分发、公众号草稿箱 |
| **OAuth X** | X（Twitter）OAuth2 回调；可挂根路径或反代前缀（如 `/tail`） |

完整字段以 **`/docs`（Swagger）** 或 **`/redoc`** 为准。

## API 鉴权（云端部署）

设置环境变量 **`LIB_API_KEY`** 后，除 `GET /`、`GET /health`、`/docs`、`/redoc`、`/openapi.json`、`/oauth2/*` 外，**所有 HTTP 路由**须携带与之一致的密钥：

- 请求头 `X-API-Key: <LIB_API_KEY>`，或
- 请求头 `Authorization: Bearer <LIB_API_KEY>`

未设置 `LIB_API_KEY` 时不启用鉴权（便于本地开发）。调用方（Cursor Automation / Agent）须在运行环境配置同名变量 **`LIB_API_KEY`**。

---

## crypto-mcp（前缀 `/crypto-mcp`）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/crypto-mcp/time` | 上海时区/北京时间 |
| `GET` | `/crypto-mcp/gainers` | U 本位 24h 涨幅榜（`limit`、`min_quote_volume`、`include_1000`） |
| `GET` | `/crypto-mcp/bundle?symbol=` | 技术 + 市场 + 情绪聚合 |
| `GET` | `/crypto-mcp/sentiment?symbol=` | 情绪聚合（RSS；`grok_analysis` 恒为空，兼容字段） |
| `GET` | `/crypto-mcp/news?symbol=` | 指定币种新闻聚合 |
| `GET` | `/crypto-mcp/charts?symbol=` | 生成 2h/4h K 线元信息 |
| `GET` | `/crypto-mcp/charts/image?...` | 返回 K 线 **PNG** |
| `GET` | `/crypto-mcp/all?symbol=` | 时间 + bundle + K 线汇总 |
| `POST` | `/crypto-mcp/distribute` | 表单 `symbol`、`text`、`image`（必填）；**服务端先 standard 清洗再发往 TG / X / Square** |

**`POST /crypto-mcp/distribute` 要点**

- 带图发帖时 `image` 必填（常见为 1 张横版 JPEG）。
- **BTC**（`BTC` / `BTCUSDT`）在 X 渠道自动引用上一条 BTC 帖（`data/x_btc_last_post_id.txt`，无需 env）。
- 其它币可选 `x_reply_to_previous=true`（读 `X_LAST_POST_ID`）。
- **Square**：有图时 OpenAPI 预签名上传 → 图文短帖；`#话题` 与 `$COIN` 写在 `text` 正文内。Key：`SQUARE_OPENAPI_KEY` 或 `BINANCE_SQUARE_OPENAPI_KEY`。

新闻源：免费 RSS（含 PANews、Odaily 等）、Sharpe 公共接口、ChainFeeds 筛选源；按 symbol 关键词边界匹配，减少短代码误报。

---

## maternal-mcp（前缀 `/maternal-mcp`）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/maternal-mcp/distribute` | 表单 `title`、`text`、`image`（必填）；**服务端先 standard 清洗再发 TG** |
| `POST` | `/maternal-mcp/wechat-draft` | 表单 `title`、`content_html`、`image`（必填）；**服务端先 standard 清洗再入草稿** |

凭据：`TG_BOT_TOKEN` / `TG_CHAT_ID`；公众号 `WECHAT_APP_ID` / `WECHAT_APP_SECRET`。

---

## 图片元数据清洗 `/images/clean`

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/images/clean/health` | 返回当日缓存的 `ffmpeg` / `exiftool` 探测结果（GMT+8 **每日 0 点**刷新，非每次请求探测） |
| `POST` | `/images/clean` | 表单 `image`（必填）、`mode=standard`（默认）、`quality=92`（1–100） |

**分发兜底**：`POST /crypto-mcp/distribute`、`POST /maternal-mcp/distribute`、`POST /maternal-mcp/wechat-draft` 上传的配图会在服务端再次执行 `mode=standard` 清洗后才发出（与 flow 侧清洗双重保障）。

**`mode=standard`（默认）**：4-pass（Pass 1 ffmpeg + `gblur=sigma=0.3` → exiftool → ffmpeg → exiftool），输出 **JPEG**（横版约 1536×1024，±2%）。  
**`mode=strip-only`**：仅剥离元数据，保留原格式。

响应头：`X-Image-Clean-Ok`、`X-Image-Clean-Report`（JSON 报告）。

典型用法：客户端在 AI 生图（如 1536×1024 PNG）成功后上传原图，取回清洗 JPEG 再分发或落盘。**本接口不处理 SynthID 隐形水印扩散**（需另行部署专用 GPU 去水印服务）。

Docker 镜像须含 **`ffmpeg`** 与 **`libimage-exiftool-perl`**（见 Dockerfile）。

---

## 环境变量

勿将真实密钥提交 Git。常见变量（是否必填以代码为准）：

**API 鉴权**  
- `LIB_API_KEY`：非空时启用全站 API Key 校验（云端部署推荐）；调用方使用同名环境变量

**币安**  
- `BINANCE_API_KEY` / `BINANCE_API_SECRET`

**分发（`distribution_service`）**  
- Telegram：`TG_BOT_TOKEN`、`TG_CHAT_ID`（兼容 `TELEGRAM_*`）  
- X OAuth2：`X_CLIENT_ID`、`X_CLIENT_SECRET`、`X_REDIRECT_URI`、`X_OAUTH2_TOKEN` 或 access/refresh token  
- X OAuth1 媒体：`X_CONSUMER_KEY`、`X_CONSUMER_SECRET`、`X_ACCESS_TOKEN`、`X_ACCESS_TOKEN_SECRET`  
- X 引用：`X_LAST_POST_ID`（非 BTC 且 `x_reply_to_previous=true`）  
- Square：`SQUARE_OPENAPI_KEY` 或 `BINANCE_SQUARE_OPENAPI_KEY`

**公众号草稿**  
- `WECHAT_APP_ID` / `WECHAT_APP_SECRET`

---

## 本地运行

**依赖**：Python 3.11+；`requirements.txt`（FastAPI、uvicorn、TA-Lib 绑定、matplotlib、aiohttp、xdk 等）。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

部署时通过环境变量配置币安与分发渠道密钥。反代示例：将 `https://your-domain/tail` 指到本服务时，OAuth 回调为 `https://your-domain/tail/oauth2/callback`（亦支持无前缀根路径，见 `app/routers/oauth_x.py`）。

---

## Docker

`docker-compose.yml` 构建本目录 Dockerfile，暴露 **8000**，挂载 `./data`、`./image`、`.env`；`TZ` 默认 `Asia/Shanghai`。`networks.nginx_network` 为 `external: true`，需预先创建或改为 `bridge`。

首次构建会从上游编译 **C 库 ta-lib**，并安装 **ffmpeg + exiftool**，耗时较长。

---

## 数据目录

| 路径 | 用途 |
|------|------|
| `data/` | 运行期数据（X OAuth token、BTC 引用帖 ID 等） |
| `image/` | K 线等生成图片落盘（`charts/image` 按规则查找当日 PNG） |

---

## 最小可跑清单

至少配置：**服务监听**、业务所需的 **币安密钥**、所用分发渠道的 **token**（TG / X / Square / 公众号按需）。`GET /images/clean/health` 仅作运维查看当日 deps 缓存，flow 执行不依赖该接口。
