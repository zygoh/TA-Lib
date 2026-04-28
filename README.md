# TA-Lib

本仓库是 **币安 U 本位合约数据与技术指标的 HTTP 服务**，在单一 FastAPI 应用中同时提供：技术指标计算、K 线相关图片、与交易/账户代理相关的接口、以及为 **MCP/Agent** 准备的 **`crypto-mcp` 聚合能力**（行情包、情绪、K 线出图、统一分发）。服务启动时还会按时间表运行 **Grok 情绪拉取** 与 **Cursor Agent 日更任务**（驱动工作区中 `zygo-skills` 的 `crypto-post-flow`）。

> 注意：本项目的 Python 包名会安装 **官方技术分析库** `TA-Lib`；仓库名与此相同，在文档中若写「本服务」多指本 FastAPI 应用，若写「ta-lib 库」多指 C 库及其 Python 绑定。

## 主要能力概览

| 模块 | 作用 |
|------|------|
| **健康检查** | `GET /`、`GET /health` 返回服务状态与时间戳。 |
| **技术指标** | 对币安期货 K 线计算指标（支持多种周期；见 `POST /calculate` 等）。 |
| **交易与账户** | 在配置币安只读/交易密钥的前提下，拉取账户信息、处理交易信号等。 |
| **图片相关** | 与 K 线/图表展示相关的图像能力（与 `Pillow`、图表服务配合）。 |
| **crypto-mcp** | 给 Agent 用的聚合接口：时间、一揽子数据、情绪、2h/4h K 线生成与取图、以及 **多平台帖子分发**。 |
| **OAuth X** | 为 X（Twitter）OAuth2 等流程提供回调与路由；可按部署习惯挂在根路径或带前缀（如与反代 `.../tail` 配合）。 |
| **后台定时任务** | 启动时：Grok 按北京时间对 ETH/BTC 读提示词跑情绪；Cursor Agent 在配置的时间对 ETH/BTC 触发日更 flow。 |

更细的接口请打开服务自带的 **`/docs`（Swagger）** 或 **`/redoc`**。

## 与 zygo-skills 的关系

- **`zygo-skills`** 仓库中 **`skills/crypto-post-flow/SKILL.md`** 是 **日更内容编排的单一事实来源**（初稿 → 压缩 → 配图 → 分发）。
- 本服务中的 **`app/services/cursor_agent_scheduler.py`** 使用 **Cursor Cloud Agent API**（`https://api.cursor.com/v0/agents`），在设定时刻向 **指定 Git 仓库** 发起任务，**提示 Agent 按上述 flow 文件执行**（并传入 `symbol=BTC` 或 `ETH`）。
- 子技能里的 **crypto-analyst** 会调用本服务的 `GET /crypto-mcp/all`、以及 K 线图的直链等（具体 Base URL 以子技能内文档为准，例如可部署在 `https://.../tail` 后挂载）。

因此：部署时通常把 **本仓库** 与 **父级工作区**（含 `zygo-skills`）**指向同一套远端仓库/分支**，并在环境变量中配置好 **Cursor 的仓库、分支、API Key 与排程时间**。

## 重要端点（crypto-mcp 前缀为 `/crypto-mcp`）

以下便于对接 Agent 与运维排查（具体字段以 Pydantic 模型与 OpenAPI 为准）：

- **`GET /crypto-mcp/time`** — 上海时区/北京时间信息。
- **`GET /crypto-mcp/bundle?symbol=...`** — 技术 + 市场 + 情绪等聚合。
- **`GET /crypto-mcp/sentiment?symbol=...`** — 情绪相关聚合。
- **`GET /crypto-mcp/sentiment/grok/status`** — Grok 后台任务状态。
- **`GET /crypto-mcp/charts?symbol=...`** — 生成 2h/4h 等 K 线输出信息（见实现）。
- **`GET /crypto-mcp/charts/image?...`** — 直接返回已生成的 **PNG 图片**（用于 Agent「看图」）。
- **`GET /crypto-mcp/all?symbol=...`** — 汇总时间与 bundle，并在流程中生成 K 线（**crypto-analyst 主调**）。
- **`POST /crypto-mcp/distribute`** — 表单提交 `symbol`、`text`、可选 `image` 文件，**统一向 Telegram、X、币安广场等渠道分发**（与 `zygo-skills` 里 `distribute-post` 能力对应）。

Grok 相关：

- 每日定时（代码内约 **ETH 07:55 / BTC 19:55 北京时间**）会读取项目内 `data/ETH_grok_prompt.md`、`data/BTC_grok_prompt.md` 作为提示内容并后台执行（与 app 中 Grok 客户端、存储配置一致）。

## 环境变量（说明性列表）

请勿把真实密钥提交到 Git。下面仅列出**常见变量名**以便部署对照（默认值与是否必填以代码为准）：

**币安**  
- `BINANCE_API_KEY` / `BINANCE_API_SECRET` — 行情、账户、交易等能力。

**Grok / xAI**  
- `XAI_API_KEY`、`XAI_MODEL`、`XAI_API_BASE_URL`（有默认基址）— Grok 情绪拉取。  
- `GROK_UPDATE_TOKEN` 等（见 `grok_store`）— 若需保护更新类接口。

**Cursor Agent 定时器**（`cursor_agent_scheduler`）  
- `CURSOR_AGENT_SCHEDULER_ENABLED` — 是否开启（默认可为开启；以代码为准）。  
- `CURSOR_API_KEY` — Cursor API 凭据。  
- `CURSOR_REPOSITORY` — 仓库（需含 `zygo-skills` 所在仓库 URL）。  
- `CURSOR_REF` — 分支，默认 `main`。  
- `CURSOR_MODEL` — 可选模型。  
- `CURSOR_SCHEDULE_TZ` — 时区，默认 `Asia/Shanghai`。  
- `CURSOR_ETH_TIME` / `CURSOR_BTC_TIME` — 每日触发时刻，默认约 `08:01` 与 `20:01`（**与 Grok 的 07:55/19:55 是两套不同任务**）。  

**Telegram 管理员通知**（仅在「调度器调用 Cursor API 失败」时发）  
- `TG_BOT_TOKEN` 或 `TELEGRAM_BOT_TOKEN`  
- `TG_ADMIN_CHAT_ID` 或 `TG_CHAT_ID` 或 `TELEGRAM_CHAT_ID`  

**分发（`distribution_service`）**  
- Telegram：`TG_BOT_TOKEN`、`TG_CHAT_ID` 等（与上部分可复用，语义以发送渠道为准）。  
- X / OAuth2：`X_CLIENT_ID`、`X_CLIENT_SECRET`、`X_REDIRECT_URI`、`X_OAUTH2_TOKEN` 或 `X_OAUTH2_ACCESS_TOKEN` / `X_OAUTH2_REFRESH_TOKEN`；以及 OAuth1 媒体上传相关 `X_CONSUMER_KEY`、`X_CONSUMER_SECRET`、`X_ACCESS_TOKEN`、`X_ACCESS_TOKEN_SECRET` 等。  
- 币安广场：`SQUARE_OPENAPI_KEY`。  
- 其他：`X_OAUTH1_CALLBACK` 等见代码。

## 本地运行与容器

**依赖**：Python 3.11+ 推荐；`requirements.txt` 中含 FastAPI、uvicorn、TA-Lib 绑定、matplotlib、aiohttp、xdk 等。

```bash
cd TA-Lib
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 需设置至少币安/业务所需环境变量
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

**Docker**：`docker-compose.yml` 会构建本目录 Dockerfile，暴露 **8000**，挂载 `./data`、`./image` 与 `.env`；`TZ` 默认 `Asia/Shanghai`。`networks.nginx_network` 为 `external: true`，需你本机/服务器已存在同名网络，或按环境改成 `bridge` 等。

构建镜像时 Dockerfile 会 **从上游编译安装 C 库 ta-lib 再装 Python 依赖**，首次构建会较慢。

## 数据与产物目录

- **`data/`** — 如 `BTC_grok_prompt.md`、`ETH_grok_prompt.md`、Grok/情绪类文本等；Grok 定时任务会读其中的提示词文件。  
- **`image/`** — K 线等生成图片的落盘位置（`charts/image` 会按规则查找当日 PNG）。  

## 相关仓库

- **zygo-skills**：日更 **crypto-post-flow** 与 **baoyu-imagine / baoyu-xhs-images** 等技能包，与本服务的 **Agent 定时器 + `crypto-mcp` 接口** 配套使用。

若你需要「最小可跑清单」，请至少配置：**服务监听、币安只读或业务所需 key、Grok 相关（若用定时情绪）、分发文案渠道对应的 token、以及（若用自动发帖）Cursor 的仓库与 API Key。**
