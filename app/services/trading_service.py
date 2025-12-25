"""
币安合约交易服务 - 完整版
1. 架构参考 Go 版本 (基于净值 Equity 计算仓位)
2. 优化 Wait 信号处理 (无信心度校验，仅日志记录)
3. 集成 python-dotenv 自动加载环境变量
4. Session 复用优化，减少 TCP 握手开销
5. 实现 exchangeInfo 精度控制，解决 Invalid quantity 问题
"""
import logging
import time
from typing import Dict, Any, List, Tuple, Optional
import aiohttp
import os
import json
import hmac
import hashlib
from urllib.parse import urlencode
import decimal

try:
    from dotenv import load_dotenv
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
except ImportError:
    print("提示: 未检测到 python-dotenv 库，将直接读取系统环境变量。建议安装: pip install python-dotenv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    default_config = {
        "binance_api_url": "https://fapi.binance.com",
        "request_timeout": 30,
    }
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
        else:
            return default_config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}，使用默认配置")
        return default_config

_config = load_config()
FIXED_LEVERAGE = 20

class BinanceClient:
    """币安 API 客户端，复用 Session 提升性能"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.exchange_info_cache: Optional[Dict[str, Any]] = None
        self.exchange_info_cache_time: float = 0
        self.cache_ttl: float = 300.0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=_config['request_timeout'])
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    def _get_api_credentials(self) -> tuple:
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            logger.warning("⚠️ 未检测到环境变量 BINANCE_API_KEY 或 BINANCE_API_SECRET")
            logger.warning("请检查是否创建了 .env 文件，或是否正确配置了系统环境变量")
        return api_key, api_secret
    
    def _sign_request(self, params: Dict) -> str:
        _, api_secret = self._get_api_credentials()
        if not api_secret:
            return ""
        query_string = urlencode(params)
        signature = hmac.new(
            api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def _send_signed_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Any:
        api_key, _ = self._get_api_credentials()
        if not api_key:
            raise ValueError("未配置API密钥，无法发送请求")

        url = f"{_config['binance_api_url']}{endpoint}"
        headers = {"X-MBX-APIKEY": api_key}
        
        if params is None:
            params = {}
        if data is None:
            data = {}
        
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._sign_request(params)

        session = await self._get_session()
        
        if method == "GET":
            async with session.request(method, url, headers=headers, params=params) as response:
                return await self._handle_response(response)
        elif method in ["POST", "DELETE"]:
            if data:
                params.update(data)
            async with session.request(method, url, headers=headers, params=params) as response:
                return await self._handle_response(response)
        else:
            raise ValueError(f"不支持的 HTTP 方法: {method}")
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Any:
        try:
            data = await response.json()
        except Exception:
            text = await response.text()
            raise Exception(f"API响应解析失败: {text}")

        if response.status != 200:
            logger.error(f"Binance API Error {response.status}: {data}")
            raise Exception(f"Binance API Error: {data.get('msg', 'Unknown error')}")
        return data
    
    async def get_exchange_info(self, symbol: str) -> Dict[str, Any]:
        """获取交易对精度信息（带缓存）"""
        current_time = time.time()
        if (self.exchange_info_cache is None or 
            current_time - self.exchange_info_cache_time > self.cache_ttl):
            url = f"{_config['binance_api_url']}/fapi/v1/exchangeInfo"
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.exchange_info_cache = {item['symbol']: item for item in data.get('symbols', [])}
                    self.exchange_info_cache_time = current_time
                else:
                    raise Exception(f"获取 exchangeInfo 失败: {response.status}")
        
        symbol_info = self.exchange_info_cache.get(symbol)
        if not symbol_info:
            raise ValueError(f"未找到交易对 {symbol} 的精度信息")
        
        filters = {f['filterType']: f for f in symbol_info.get('filters', [])}
        lot_size = filters.get('LOT_SIZE', {})
        price_filter = filters.get('PRICE_FILTER', {})
        
        return {
            'stepSize': float(lot_size.get('stepSize', '1.0')),
            'tickSize': float(price_filter.get('tickSize', '0.01')),
            'minQty': float(lot_size.get('minQty', '0.001')),
            'maxQty': float(lot_size.get('maxQty', '1000000')),
        }
    
    async def get_symbol_price(self, symbol: str) -> float:
        url = f"{_config['binance_api_url']}/fapi/v1/ticker/price"
        session = await self._get_session()
        async with session.get(url, params={"symbol": symbol}) as response:
            data = await response.json()
            return float(data.get("price", 0))
    
    async def get_funding_rate(self, symbol: str) -> float:
        url = f"{_config['binance_api_url']}/fapi/v1/premiumIndex"
        session = await self._get_session()
        async with session.get(url, params={"symbol": symbol}) as response:
            if response.status == 200:
                data = await response.json()
                return float(data.get("lastFundingRate", 0)) * 100
            return 0.0

_client = BinanceClient()

def round_step_size(quantity: float, step_size: float) -> float:
    """根据 stepSize 精度要求舍入数量（使用 Decimal 避免浮点精度问题）"""
    if step_size <= 0:
        return quantity
    qty = decimal.Decimal(str(quantity))
    step = decimal.Decimal(str(step_size))
    return float((qty // step) * step)

# --- 业务计算逻辑 (参考 Go 架构) ---

def calculate_position_limits(equity: float, is_btc_eth: bool) -> Tuple[float, float]:
    """
    基于账户净值计算仓位范围
    BTC/ETH: Min = Equity * 10, Max = Equity * 19
    Alts:    Min = Equity * 5,  Max = Equity * 19
    """
    safe_leverage = FIXED_LEVERAGE - 1.0
    
    if is_btc_eth:
        min_size = equity * 10.0
        max_size = equity * safe_leverage
    else:
        min_size = equity * 5.0
        max_size = equity * safe_leverage
        
    return min_size, max_size

def format_duration_minutes(last_update_time_ms: int) -> int:
    """计算持仓时长（分钟），基于最后更新时间"""
    current_time_ms = int(time.time() * 1000)
    duration_ms = current_time_ms - last_update_time_ms
    return int(duration_ms / 60000)

# --- 账户与持仓模块 ---

async def get_open_orders(symbol: str = None) -> List[Dict]:
    params = {}
    if symbol:
        params['symbol'] = symbol
    return await _client._send_signed_request("GET", "/fapi/v1/openOrders", params)

async def format_account_summary(account_data: Dict[str, Any], positions_count: int) -> str:
    total_wallet_balance = float(account_data.get("totalWalletBalance", 0))
    available_balance = float(account_data.get("availableBalance", 0))
    total_unrealized_pnl = float(account_data.get("totalUnrealizedProfit", 0))
    
    equity = total_wallet_balance + total_unrealized_pnl
    
    balance_ratio = (available_balance / equity * 100) if equity > 0 else 0
    pnl_percent = (total_unrealized_pnl / total_wallet_balance * 100) if total_wallet_balance > 0 else 0
    margin_ratio = ((equity - available_balance) / equity * 100) if equity > 0 else 0
    
    return (f"净值{equity:.2f} | "
            f"余额{available_balance:.2f} ({balance_ratio:.1f}%) | "
            f"盈亏{pnl_percent:+.2f}% | "
            f"保证金{margin_ratio:.1f}% | "
            f"持仓{positions_count}个")

async def format_positions(positions_data: List[Dict[str, Any]], open_orders: List[Dict]) -> List[Dict[str, Any]]:
    formatted_positions = []
    
    for pos in positions_data:
        symbol = pos.get("symbol", "")
        position_amt = float(pos.get("positionAmt", 0))
        entry_price = float(pos.get("entryPrice", 0))
        mark_price = float(pos.get("markPrice", 0))
        leverage = int(pos.get("leverage", FIXED_LEVERAGE))
        liquidation_price = float(pos.get("liquidationPrice", 0))
        last_update_time = int(pos.get("updateTime", time.time() * 1000))
        
        side = "LONG" if position_amt > 0 else "SHORT"
        
        pnl_percent = ((mark_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        if side == "SHORT":
            pnl_percent = -pnl_percent
        
        margin = abs(position_amt * entry_price / leverage)
        duration_minutes = format_duration_minutes(last_update_time)
        funding_rate = await _client.get_funding_rate(symbol)

        stop_loss = None
        take_profit = None
        
        symbol_orders = [o for o in open_orders if o['symbol'] == symbol]
        for order in symbol_orders:
            o_type = order.get('type')
            stop_price = float(order.get('stopPrice', 0))
            if o_type in ['STOP_MARKET', 'STOP']:
                stop_loss = stop_price
            elif o_type in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT']:
                take_profit = stop_price

        sl_str = f"{stop_loss:.4f}" if stop_loss else "未设置"
        tp_str = f"{take_profit:.4f}" if take_profit else "未设置"
        
        display_string = (
            f"{symbol} {side} | "
            f"入场价{entry_price:.4f} 当前价{mark_price:.4f} | "
            f"盈亏{pnl_percent:+.2f}% | 杠杆{leverage}x | "
            f"保证金{margin:.0f} | 强平价{liquidation_price:.4f} | "
            f"持仓时长{duration_minutes}分钟 | "
            f"止损{sl_str} 止盈{tp_str} | "
            f"资金费率{funding_rate:.4f}%"
        )

        formatted_positions.append({
            "symbol": symbol,
            "side": side,
            "entry_price": round(entry_price, 4),
            "current_price": round(mark_price, 4),
            "pnl_percent": round(pnl_percent, 2),
            "leverage": leverage,
            "margin": round(margin, 2),
            "liquidation_price": round(liquidation_price, 4),
            "duration_minutes": duration_minutes,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "funding_rate": round(funding_rate, 4),
            "display_string": display_string
        })
    
    return formatted_positions

async def get_positions() -> List[Dict[str, Any]]:
    positions_data = await _client._send_signed_request("GET", "/fapi/v2/positionRisk")
    active_positions = [p for p in positions_data if float(p.get("positionAmt", 0)) != 0]
    
    try:
        all_open_orders = await get_open_orders()
    except Exception as e:
        logger.warning(f"获取挂单失败: {e}")
        all_open_orders = []

    return await format_positions(active_positions, all_open_orders)

async def get_account_info() -> Dict[str, Any]:
    account_data = await _client._send_signed_request("GET", "/fapi/v2/account")
    return await format_account_response(account_data)

async def format_account_response(account_data: Dict[str, Any]) -> Dict[str, Any]:
    # 计算净值 (Equity)
    total_wallet_balance = float(account_data.get("totalWalletBalance", 0))
    total_unrealized_pnl = float(account_data.get("totalUnrealizedProfit", 0))
    equity = total_wallet_balance + total_unrealized_pnl
    
    positions = await get_positions()
    positions_count = len(positions)
    timestamp = int(time.time() * 1000)
    
    # 动态计算仓位范围 (Equity Based)
    alt_min, alt_max = calculate_position_limits(equity, is_btc_eth=False)
    btc_min, btc_max = calculate_position_limits(equity, is_btc_eth=True)
    
    # 展示用的安全杠杆
    safe_leverage_display = FIXED_LEVERAGE - 1
    
    single_coin_position = (
        f"山寨{int(alt_min)}-{int(alt_max)} U({safe_leverage_display}x杠杆) | "
        f"BTC/ETH {int(btc_min)}-{int(btc_max)} U({safe_leverage_display}x杠杆)"
    )
    
    account_summary = await format_account_summary(account_data, positions_count)
    
    return {
        "timestamp": timestamp,
        "single_coin_position": single_coin_position,
        "account_summary": account_summary,
        "positions": positions
    }

async def get_account_markdown() -> str:
    """获取账户信息的 Markdown 格式字符串（供 Agent 使用）"""
    account_data = await _client._send_signed_request("GET", "/fapi/v2/account")
    formatted = await format_account_response(account_data)
    
    markdown = f"""账户详情：

当前时间戳: {formatted['timestamp']}

单币仓位: {formatted['single_coin_position']}

账户: {formatted['account_summary']}

## 当前持仓

"""
    
    positions = formatted['positions']
    if positions:
        for idx, pos in enumerate(positions, 1):
            stop_loss = f"{pos['stop_loss']:.4f}" if pos.get('stop_loss') else "未设置"
            take_profit = f"{pos['take_profit']:.4f}" if pos.get('take_profit') else "未设置"
            
            markdown += (f"{idx}. {pos['symbol']} {pos['side']} | "
                        f"入场价{pos['entry_price']:.4f} 当前价{pos['current_price']:.4f} | "
                        f"盈亏{pos['pnl_percent']:+.2f}% | 杠杆{pos['leverage']}x | "
                        f"保证金{int(pos['margin'])} | 强平价{pos['liquidation_price']:.4f} | "
                        f"持仓时长{pos['duration_minutes']}分钟 | "
                        f"止损{stop_loss} 止盈{take_profit} | "
                        f"资金费率{pos['funding_rate']:.4f}%\n")
    else:
        markdown += "暂无持仓\n"
    
    return markdown

# --- 信号执行模块 ---

async def set_leverage(symbol: str, leverage: int):
    try:
        await _client._send_signed_request("POST", "/fapi/v1/leverage", data={
            "symbol": symbol,
            "leverage": leverage
        })
    except Exception as e:
        error_msg = str(e)
        if "No need to change" in error_msg or "无需修改" in error_msg:
            logger.debug(f"杠杆已为 {leverage}x，无需修改: {symbol}")
        else:
            logger.warning(f"设置杠杆失败 {symbol}: {e}")

async def execute_trade(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行具体的下单操作
    注意：这里不再处理 wait/hold，这些状态已在 process_trading_signals 中过滤
    """
    symbol = signal.get("symbol")
    action = signal.get("action")
    leverage = FIXED_LEVERAGE
    
    try:
        if action in ["open_long", "open_short"]:
            await set_leverage(symbol, leverage)
            
            current_price = await _client.get_symbol_price(symbol)
            position_size_usd = float(signal.get("position_size_usd", 0))
            if position_size_usd <= 0:
                raise ValueError("下单金额无效")
            
            exchange_info = await _client.get_exchange_info(symbol)
            step_size = exchange_info['stepSize']
            
            quantity = position_size_usd / current_price
            quantity = round_step_size(quantity, step_size)
            
            if quantity < exchange_info['minQty']:
                raise ValueError(f"数量 {quantity} 小于最小数量 {exchange_info['minQty']}")
            if quantity > exchange_info['maxQty']:
                raise ValueError(f"数量 {quantity} 大于最大数量 {exchange_info['maxQty']}")
            
            side = "BUY" if action == "open_long" else "SELL"
            
            order_res = await _client._send_signed_request("POST", "/fapi/v1/order", data={
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity
            })
            
            sl_price = signal.get("stop_loss")
            tp_price = signal.get("take_profit")
            close_side = "SELL" if side == "BUY" else "BUY"
            
            if sl_price:
                await _client._send_signed_request("POST", "/fapi/v1/order", data={
                    "symbol": symbol,
                    "side": close_side,
                    "type": "STOP_MARKET",
                    "stopPrice": sl_price,
                    "closePosition": "true",
                    "workingType": "MARK_PRICE"
                })
            
            if tp_price:
                await _client._send_signed_request("POST", "/fapi/v1/order", data={
                    "symbol": symbol,
                    "side": close_side,
                    "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": tp_price,
                    "closePosition": "true",
                    "workingType": "MARK_PRICE"
                })
                
            return {"status": "success", "order_id": order_res.get("orderId"), "msg": "开仓成功"}

        elif action == "adjust_stops":
            await _client._send_signed_request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol})
            
            positions = await get_positions()
            target_pos = next((p for p in positions if p["symbol"] == symbol), None)
            
            if not target_pos:
                return {"status": "error", "msg": "无持仓，无法调整"}
            
            pos_side = target_pos["side"]
            close_side = "SELL" if pos_side == "LONG" else "BUY"
            
            sl_price = signal.get("stop_loss")
            tp_price = signal.get("take_profit")
            
            if sl_price:
                await _client._send_signed_request("POST", "/fapi/v1/order", data={
                    "symbol": symbol,
                    "side": close_side,
                    "type": "STOP_MARKET",
                    "stopPrice": sl_price,
                    "closePosition": "true",
                    "workingType": "MARK_PRICE"
                })
            
            if tp_price:
                await _client._send_signed_request("POST", "/fapi/v1/order", data={
                    "symbol": symbol,
                    "side": close_side,
                    "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": tp_price,
                    "closePosition": "true",
                    "workingType": "MARK_PRICE"
                })
            
            return {"status": "success", "msg": "止损止盈已更新"}

        elif action in ["close_long", "close_short"]:
            try:
                await _client._send_signed_request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol})
            except Exception:
                pass
            
            risk_data = await _client._send_signed_request("GET", "/fapi/v2/positionRisk", params={"symbol": symbol})
            amt = abs(float(risk_data[0].get("positionAmt", 0)))
            
            if amt > 0:
                exchange_info = await _client.get_exchange_info(symbol)
                step_size = exchange_info['stepSize']
                amt = round_step_size(amt, step_size)
                
                side = "SELL" if "long" in action else "BUY"
                await _client._send_signed_request("POST", "/fapi/v1/order", data={
                    "symbol": symbol,
                    "side": side,
                    "type": "MARKET",
                    "quantity": amt,
                    "reduceOnly": "true"
                })
                return {"status": "success", "msg": "已平仓"}
            return {"status": "skipped", "msg": "当前无持仓"}

        else:
            return {"status": "error", "msg": f"未知动作: {action}"}

    except Exception as e:
        logger.error(f"交易执行异常 {symbol}: {e}")
        return {"status": "error", "msg": str(e)}

async def process_trading_signals(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    处理交易信号入口
    优化点：Wait信号无信心度检查，直接跳过处理
    """
    results = []
    
    for signal in signals:
        action = signal.get("action")
        symbol = signal.get("symbol", "")
        
        # 1. 优先处理 Wait/Hold (免检逻辑)
        if action in ["wait", "hold"]:
            reason = signal.get("reasoning", "No reasoning provided")
            logger.info(f"Signal [WAIT/HOLD]: {symbol} - {reason}")
            results.append({
                "symbol": symbol,
                "action": action,
                "status": "skipped",
                "message": f"观望: {reason}"
            })
            continue # 直接跳过，不进入后续验证和执行

        # 2. 开仓信号校验 (包含信心度检查)
        if action in ["open_long", "open_short"]:
            confidence = signal.get("confidence", 0)
            if confidence < 90:
                results.append({
                    "symbol": symbol,
                    "action": action,
                    "status": "rejected",
                    "reason": f"信心不足 {confidence} < 90"
                })
                continue
            
            req_fields = ["leverage", "position_size_usd", "stop_loss", "take_profit", "risk_usd"]
            if any(signal.get(f) is None for f in req_fields):
                results.append({
                    "symbol": symbol,
                    "action": action,
                    "status": "rejected",
                    "reason": "开仓参数缺失"
                })
                continue

        # 3. 调仓信号校验
        elif action == "adjust_stops":
            if not signal.get("stop_loss") or not signal.get("take_profit"):
                results.append({
                    "symbol": symbol,
                    "action": action,
                    "status": "rejected",
                    "reason": "调整缺少SL/TP参数"
                })
                continue
        
        # 4. 执行交易
        exec_res = await execute_trade(signal)
        results.append({
            "symbol": symbol,
            "action": action,
            "status": exec_res.get("status"),
            "message": exec_res.get("msg"),
            "order_id": exec_res.get("order_id")
        })
    
    return {
        "processed_count": len(results),
        "results": results
    }