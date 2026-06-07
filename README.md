# TA-Lib

本仓库是 **币安 U 本位合约数据与技术指标的 HTTP 服务**，在单一 FastAPI 应用中同时提供：技术指标计算、K 线相关图片、与交易/账户代理相关的接口、以及为 **MCP/Agent** 准备的 **`crypto-mcp` 聚合能力**（行情包、基于 RSS 的情绪/新闻聚合、K 线出图、统一分发）。**不再内置对 xAI Grok 的调用。**

> 注意：本项目的 Python 包名会安装 **官方技术分析库** `TA-Lib`；仓库名与此相同，在文档中若写「本服务」多指本 FastAPI 应用，若写「ta-lib 库」多指 C 库及其 Python 绑定。

## 主要能力概览

| 模块 | 作用 |
|------|------|
| **健康检查** | `GET /`、`GET /health` 返回服务状态与时间戳。 |
| **技术指标** | 对币安期货 K 线计算指标（支持多种周期；见 `POST /calculate` 等）。 |
| **交易与账户** | 在配置币安只读/交易密钥的前提下，拉取账户信息、处理交易信号等。 |
| **图片相关** | 与 K 线/图表展示相关的图像能力（与 `Pillow`、图表服务配合）。 |
| **crypto-mcp** | 给 Agent 用的聚合接口：时间、币安 U 本位涨幅榜、一揽子数据、情绪/新闻（免费 RSS + 公共新闻接口）、2h/4h K 线生成与取图、以及 **多平台帖子分发**。 |
| **OAuth X** | 为 X（Twitter）OAuth2 等流程提供回调与路由；可按部署习惯挂在根路径或带前缀（如与反代 `.../tail` 配合）。 |

更细的接口请打开服务自带的 **`/docs`（Swagger）** 或 **`/redoc`**。

## 与 zygo-skills 的关系

- **币圈日更**：`zygo-skills/skills/crypto/crypto-post-flow/SKILL.md` 是编排真源（选币认领 → 初稿 → 卦象加持 → Stage 2.0 → 并行压缩/配图（≤4 次重试）→ Stage 2.5 → 带图分发 → 更新 `MEMORIES.md`）。`distribute-post` 对应 **`POST /crypto-mcp/distribute`**（`image` 必传；TG + X + Square）。Stage 4 仅提交 `skills/crypto/crypto-post-flow/MEMORIES.md`（图片风格记忆）。
- **母婴科普**：`zygo-skills/skills/maternal-post-flow/SKILL.md` 是编排真源（选题 → 权威取证 → 合规前置 → 并行清洗/配图 → Stage 2.5 双重校验 → **只发 TG** → 更新 `MEMORIES.md`）。`maternal-distribute` 对应 **`POST /maternal-mcp/distribute`**（`title` + `text` + `image` 必传；`digest` 可选）。Stage 4 仅提交 `skills/maternal-post-flow/MEMORIES.md`（选题 + 配图风格）。
- **crypto-analyst** 调用 `GET /crypto-mcp/all` 及 K 线图直链等（Base URL 以子技能为准，生产示例 `https://do2ge.com/tail`）。
- 推荐在 **Cursor Cloud Automation** 中运行各 flow；本服务**不再**内置 Cursor Agent 定时器。

部署时通常把 **本仓库** 与 **zygo-skills** 指向可用的远端分支，以便 Agent 技能与本服务 `crypto-mcp` / `maternal-mcp` 协同。人类可读设计见工作区 `docs/DESIGN-symbol-selection-pipeline.md`、`docs/DESIGN-flow-image.md`、`docs/DESIGN-maternal-post-flow.md`。

## 重要端点（crypto-mcp 前缀为 `/crypto-mcp`）

以下便于对接 Agent 与运维排查（具体字段以 Pydantic 模型与 OpenAPI 为准）：

- **`GET /crypto-mcp/time`** — 上海时区/北京时间信息。
- **`GET /crypto-mcp/gainers?limit=20&min_quote_volume=5000000&include_1000=true`** — 币安 U 本位合约 24h 涨幅榜；按 `priceChangePercent` 降序返回，默认过滤低成交额噪音。
- **`GET /crypto-mcp/bundle?symbol=...`** — 技术 + 市场 + 情绪等聚合。
- **`GET /crypto-mcp/sentiment?symbol=...`** — 情绪相关聚合（RSS；响应中 `grok_analysis` 恒为空字符串，仅为兼容保留）。
- **`GET /crypto-mcp/news?symbol=...`** — 指定币种免费新闻聚合（与 `sentiment` 同源，便于从涨幅榜挑选 symbol 后单独查询新闻）。
- **`GET /crypto-mcp/charts?symbol=...`** — 生成 2h/4h 等 K 线输出信息（见实现）。
- **`GET /crypto-mcp/charts/image?...`** — 直接返回已生成的 **PNG 图片**（用于 Agent「看图」）。
- **`GET /crypto-mcp/all?symbol=...`** — 汇总时间与 bundle（若 symbol 在 12h 热榜，含 `bundle.hot_board_supplement`），并生成 K 线（**crypto-analyst 主调**）。
- **`POST /crypto-mcp/subscription-inbox/seed`** — 仅开发用假数据；**生产验收** `uv run python scripts/test_pipeline_api.py`（默认 `https://do2ge.com/tail`，仅 §3 选币管线 GET；`--consume-inbox` 才测 POST consume）。
- **`POST /crypto-mcp/subscription-inbox/consume`** — 取出 @wizzalert 待处理 raw 并**物理删除**（ingest skill）。
- **`POST /crypto-mcp/hot-board/upsert`** — 热榜写入。ingest：`symbol` + `source=wizz_alert` + **`alert_reason`（必填）**；Merger：`source=merger_analyzer`（`merger` 仅库内评分，不返回 Agent）。
- **`GET /crypto-mcp/hot-board/picker-snapshot`** — 热榜候选；`include_pick_ta=true` 时服务端并发返回每条 `pick_ta`（1h/2h/4h+市场，无 RSS）。`hot-board-pick` 用 `max_symbols=100&include_pick_ta=true`；排除 2h `pick_cooldown`。
- **`POST /crypto-mcp/pick-slot`** — `hot-board-pick` 提交选中币 + `candidate_symbols`（落选写 2h 冷却）。
- **`GET /crypto-mcp/pick-slot?consume=true`** — `crypto-post-flow` Stage 0 认领待发帖单槽。
- **`GET /crypto-mcp/futures-symbols`** — 币安 U 本位 TRADING 合约列表（ingest 校验）。
- **`POST /crypto-mcp/distribute`** — 表单提交 `symbol`、`text`、`image`（HTTP 层可选；**`crypto-post-flow` 经 `distribute-post` 调用时必传**），**统一向 Telegram、X、币安 Square 等渠道分发**。**BTC**（`BTC` / `BTCUSDT`）在 X 渠道会**自动**引用上一条 BTC 帖（服务端 `data/x_btc_last_post_id.txt` 自动读写，无需配置）；其它币可选 `x_reply_to_previous=true`（读 `X_LAST_POST_ID`）。

**maternal-mcp 前缀为 `/maternal-mcp`**（母婴科普 flow 专用，只发 Telegram）：

- **`POST /maternal-mcp/distribute`** — 表单提交 `title`、`digest`（可选）、`text`、`image`（**必填**）。服务端顺序：① `sendPhoto`（封面 + title/digest 说明）→ ② `sendMessage`（完整正稿）。**只走 Telegram**，不碰 X / Square / 公众号。凭据读 `.env` 的 `TG_BOT_TOKEN` / `TG_CHAT_ID`（与上表分发节相同）。对应 **`zygo-skills`** 的 `maternal-post-flow` → `maternal-distribute`。

  **Binance Square 行为**（对齐 [Binance square-post skill](https://github.com/binance/binance-skills-hub/tree/main/skills/binance/square-post)）：
  - 有 `image`：先走 OpenAPI 预签名上传 → 轮询处理 → 以 `contentType=1` + `imageList` 发图文短帖（`crypto-post-flow` 固定传 **1 张**）。
  - 无 `image`：纯文本短帖（仅 `bodyTextOnly`；**日更 flow 不走此路径**）。
  - **标签**：`#话题` 与 `$COIN` 须写在 `text` 正文内，由 Square 服务端解析；接口无单独 tags 字段。
  - OpenAPI Key：`SQUARE_OPENAPI_KEY` 或 `BINANCE_SQUARE_OPENAPI_KEY`（请求头 `X-Square-OpenAPI-Key`）。

新闻源说明：当前仅使用免费来源，包括站点 RSS、PANews RSS、Odaily RSS、Sharpe 公共新闻接口，以及从 ChainFeeds 源目录中筛选出的可用免费媒体/Newsletter RSS；不依赖 CryptoPanic / CoinGecko News 等付费 API。动态币种会按 symbol 生成关键词并做边界匹配，减少 `SUI`、`ARB` 等短代码误报。

## 环境变量（说明性列表）

请勿把真实密钥提交到 Git。下面仅列出**常见变量名**以便部署对照（默认值与是否必填以代码为准）：

**币安**  
- `BINANCE_API_KEY` / `BINANCE_API_SECRET` — 行情、账户、交易等能力。

**选币管线（Telethon 监听 + 热榜，见工作区 `docs/DESIGN-symbol-selection-pipeline.md`）**  
- `TG_LISTEN_API_ID` / `TG_LISTEN_API_HASH` — Telethon 用户号（监听 @wizzalert，仅写收件箱）。  
- `TG_LISTEN_SESSION_PATH` — 可选，session 文件前缀，默认 `data/telegram_listen`。  
- `WIZZ_ALERT_CHANNEL` — 默认 `wizzalert`。  
- **Merger VOS**：后台每 15min 扫盘入热榜（规则见 `app/services/merger_analyzer.py` 顶部常量）。

**分发（`distribution_service`）**  
- Telegram：`TG_BOT_TOKEN`、`TG_CHAT_ID` 等（与上部分可复用，语义以发送渠道为准）。  
- X / OAuth2：`X_CLIENT_ID`、`X_CLIENT_SECRET`、`X_REDIRECT_URI`、`X_OAUTH2_TOKEN` 或 `X_OAUTH2_ACCESS_TOKEN` / `X_OAUTH2_REFRESH_TOKEN`；以及 OAuth1 媒体上传相关 `X_CONSUMER_KEY`、`X_CONSUMER_SECRET`、`X_ACCESS_TOKEN`、`X_ACCESS_TOKEN_SECRET` 等。  
- X 引用转帖：`X_LAST_POST_ID`（非 BTC 且 `x_reply_to_previous=true` 时读取；每次 X 发帖成功后更新）。**BTC 专用链**：无 env，由服务写入 `data/x_btc_last_post_id.txt`（仅 BTC/BTCUSDT 发帖时更新，与 ETH 等互不影响）。  
- 币安广场：`SQUARE_OPENAPI_KEY` 或 `BINANCE_SQUARE_OPENAPI_KEY`（Square OpenAPI；与 [square-post skill](https://github.com/binance/binance-skills-hub/tree/main/skills/binance/square-post) 一致）。
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

- **`data/`** — 运行期数据；选币管线 SQLite 默认 **`data/pipeline/pipeline.db`**（收件箱 + 热榜）。历史 `*_grok_prompt.md` 可删可留，服务端不再读取。  
- **`image/`** — K 线等生成图片的落盘位置（`charts/image` 会按规则查找当日 PNG）。  

## 相关仓库

- **zygo-skills**：**crypto-post-flow**（`crypto-mcp`）与 **maternal-post-flow**（`maternal-mcp`）编排契约；配图均为横版 4:3 原生 `GenerateImage` 1536×1024，不裁切。

若你需要「最小可跑清单」，请至少配置：**服务监听、币安只读或业务所需 key、分发文案渠道对应的 token。**
