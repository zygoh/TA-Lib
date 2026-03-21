# TA-Lib Crypto MCP API（4 个接口）

服务地址（Base URL）：`http://localhost:8000`

统一说明：
- 全部接口前缀：`/crypto-mcp`
- `symbol`：支持 `BTC` / `BTCUSDT`（服务端会统一转换为 `BTCUSDT`）

---

## 1) Grok AI 情绪分析：`GET /crypto-mcp/sentiment/grok/{SYMBOL}`

用途：按 SYMBOL（`BTC` 或 `ETH`，也支持 `BTCUSDT` / `ETHUSDT`）读取仓库内 `data/{SYMBOL}_grok_prompt.md` 作为发给 Grok 的提示内容，后台异步调用 Grok 4.1 AI 获取市场情绪分析，立即返回。任务完成后结果写入 `data/grok_sentiment.txt`，通过状态查询接口获取进度。

### 请求

```http
GET http://localhost:8000/crypto-mcp/sentiment/grok/BTC
```

```http
GET http://localhost:8000/crypto-mcp/sentiment/grok/ETH
```

### 响应

任务已提交：

```json
{
  "ok": true
}
```

已有任务执行中：

```json
{
  "ok": false,
  "error": "已有任务正在执行中，请稍后再试"
}
```

---

## 2) Grok 任务状态查询：`GET /crypto-mcp/sentiment/grok/status`

用途：查询 Grok AI 情绪分析后台任务的执行状态。

### 请求

```http
GET http://localhost:8000/crypto-mcp/sentiment/grok/status
```

### 响应

```json
{
  "status": "completed",
  "error": null,
  "updated_at": 1710856377
}
```

`status` 取值：
- `idle`：无任务
- `running`：任务执行中
- `completed`：任务完成，结果已写入文件
- `failed`：任务失败，`error` 字段包含错误原因

---

## 3) 一站式汇总：`GET /crypto-mcp/all`

用途：一次性返回 `time` + `bundle`（不包含 charts）。

### 请求

```http
GET http://localhost:8000/crypto-mcp/all?symbol=BTC
```

### 响应（JSON）示意

```json
{
  "symbol": "BTCUSDT",
  "time": { "full": "...", "short": "...", "timestamp": 1700000000 },
  "bundle": {
    "target": "BTCUSDT",
    "technical_analysis": { "...": "..." },
    "market_analysis": { "...": "..." },
    "sentiment_analysis": { "...": "..." }
  }
}
```

---

## 4) 获取图表 PNG：`GET /crypto-mcp/charts/image`

用途：直接返回图表 PNG（二进制流）。

说明：
- `interval` 仅支持 `2h` 或 `4h`（默认 `2h`）
- 若当天图片不存在，会返回 `404`（通常需要先触发一次图表生成接口 `/crypto-mcp/charts`，或由其它流程提前生成）

### 请求

```http
GET http://localhost:8000/crypto-mcp/charts/image?symbol=BTC&interval=4h
```

### 响应

- `200`：`Content-Type: image/png`，返回 PNG 二进制
- `400`：`interval` 非 `2h/4h`
- `404`：当天图片不存在

### PowerShell 保存图片示例

```powershell
Invoke-WebRequest `
  -Method GET `
  -Uri "http://localhost:8000/crypto-mcp/charts/image?symbol=BTC&interval=4h" `
  -OutFile ".\\BTC_4h.png"
```

