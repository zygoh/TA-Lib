"""
币安技术指标计算服务 - 使用TA-Lib
包含API调用和指标计算功能，支持按日重置 VWAP 和过滤 null 数据
[优化版] 针对AI Token消耗进行了结构优化
[修改版] 修复OI接口路径，修复OI合并容差，实施安全Token优化(仅时间与量)
[增强版] VWAP增加Session模式 (固定时长重置)
"""
import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Union
from urllib.parse import urlencode

import aiohttp
import pandas as pd
import talib
from aiohttp.client_exceptions import ClientResponseError

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置文件
def load_config():
    """加载配置文件"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    default_config = {
        "binance_api_url": "https://fapi.binance.com",
        "thread_pool_size": 10,
        "request_timeout": 30,
        "retry_attempts": 3,
        "retry_delay": 1
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        for key, value in default_config.items():
            if key not in config:
                logger.warning(f"配置文件缺少 {key}，使用默认值 {value}")
                config[key] = value
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}，使用默认配置")
        return default_config

# 全局配置
_config = load_config()

# 创建全局线程池
_thread_pool = ThreadPoolExecutor(max_workers=_config["thread_pool_size"], thread_name_prefix="indicator_worker")

def ensure_float64(data: pd.Series) -> pd.Series:
    """确保数据为float64类型，TA-Lib要求"""
    if data.dtype != 'float64':
        return data.astype('float64')
    return data

def round_float(val: Union[float, int, None], precision: int = 8) -> Union[float, int, None]:
    """智能浮点数截断"""
    if pd.isna(val) or val is None:
        return None
    try:
        return round(float(val), precision)
    except Exception:
        return val


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    """
    将币安时间间隔字符串转换为pandas Timedelta对象
    
    Args:
        interval: 时间间隔字符串，如 '15m', '1h', '4h', '1d'
    
    Returns:
        pd.Timedelta对象，表示该时间间隔的时长
    
    Raises:
        ValueError: 当提供不支持的时间间隔时
    
    Examples:
        >>> interval_to_timedelta("15m")
        Timedelta('0 days 00:15:00')
        >>> interval_to_timedelta("1h")
        Timedelta('0 days 01:00:00')
        >>> interval_to_timedelta("1d")
        Timedelta('1 days 00:00:00')
    """
    INTERVAL_MAP = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "2h": pd.Timedelta(hours=2),
        "4h": pd.Timedelta(hours=4),
        "6h": pd.Timedelta(hours=6),
        "8h": pd.Timedelta(hours=8),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
    }
    
    if interval not in INTERVAL_MAP:
        raise ValueError(f"不支持的时间间隔: {interval}. 支持的间隔: {list(INTERVAL_MAP.keys())}")
    
    return INTERVAL_MAP[interval]


def validate_output_data_alignment(final_data: Dict[str, List], expected_length: int) -> None:
    """
    验证输出数据的所有列长度一致性
    
    Args:
        final_data: 最终输出的数据字典，key为列名，value为数据列表
        expected_length: 期望的数据长度（通常为DataFrame的行数）
    
    Raises:
        ValueError: 如果发现任何列的长度与期望长度不一致
    
    Examples:
        >>> data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> validate_output_data_alignment(data, 3)  # 通过，不抛异常
        
        >>> data = {'a': [1, 2, 3], 'b': [4, 5]}
        >>> validate_output_data_alignment(data, 3)  # 抛出ValueError
    """
    mismatched_columns = []
    
    for col_name, col_data in final_data.items():
        if isinstance(col_data, list):
            actual_length = len(col_data)
            if actual_length != expected_length:
                mismatched_columns.append({
                    'column': col_name,
                    'expected': expected_length,
                    'actual': actual_length,
                    'diff': actual_length - expected_length
                })
    
    if mismatched_columns:
        error_details = '\n'.join([
            f"  - {m['column']}: 期望{m['expected']}, 实际{m['actual']}, 差异{m['diff']}"
            for m in mismatched_columns
        ])
        raise ValueError(
            f"❌ 数据对齐验证失败，发现{len(mismatched_columns)}个列长度不一致:\n{error_details}"
        )


def calculate_oi_data_info(oi_series: pd.Series, total_bars: int) -> Dict[str, Any]:
    """
    计算OI（持仓量）数据的有效性信息
    
    由于币安API限制，OI数据只能获取最近30天。对于长周期数据（如1d周期500条），
    前面大部分数据的OI会被填充为0。此函数用于标识哪些数据是真实的OI，哪些是填充的0。
    
    Args:
        oi_series: OI数据序列（pandas Series）
        total_bars: 总K线数量
    
    Returns:
        包含以下字段的字典:
        - total_bars: 总K线数量
        - oi_valid_bars: 有效OI数据的数量
        - oi_valid_from_index: 有效OI数据的起始索引
        - oi_coverage_percent: OI数据覆盖率（百分比）
    
    Examples:
        >>> oi = pd.Series([100, 200, 300])
        >>> calculate_oi_data_info(oi, 3)
        {'total_bars': 3, 'oi_valid_bars': 3, 'oi_valid_from_index': 0, 'oi_coverage_percent': 100.0}
        
        >>> oi = pd.Series([0, 0, 100, 200])
        >>> calculate_oi_data_info(oi, 4)
        {'total_bars': 4, 'oi_valid_bars': 2, 'oi_valid_from_index': 2, 'oi_coverage_percent': 50.0}
    """
    # 找到第一个非0且非NaN的索引
    valid_mask = (oi_series != 0) & (~oi_series.isna())
    
    if valid_mask.any():
        # 找到第一个有效值的位置
        first_valid_idx = valid_mask.idxmax()
        valid_bars = total_bars - first_valid_idx
        coverage_percent = (valid_bars / total_bars) * 100 if total_bars > 0 else 0.0
    else:
        # 全部无效
        first_valid_idx = total_bars
        valid_bars = 0
        coverage_percent = 0.0
    
    return {
        "total_bars": int(total_bars),
        "oi_valid_bars": int(valid_bars),
        "oi_valid_from_index": int(first_valid_idx),
        "oi_coverage_percent": round(float(coverage_percent), 2)
    }


class TechnicalIndicators:
    """技术指标计算类 - 使用TA-Lib库"""

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        data = ensure_float64(data)
        return pd.Series(talib.SMA(data.values, timeperiod=period), index=data.index)

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        data = ensure_float64(data)
        return pd.Series(talib.EMA(data.values, timeperiod=period), index=data.index)

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        data = ensure_float64(data)
        return pd.Series(talib.RSI(data.values, timeperiod=period), index=data.index)

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        data = ensure_float64(data)
        upper, middle, lower = talib.BBANDS(data.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
        return {
            'upper': pd.Series(upper, index=data.index),
            'middle': pd.Series(middle, index=data.index),
            'lower': pd.Series(lower, index=data.index)
        }

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        data = ensure_float64(data)
        macd_line, signal_line, histogram = talib.MACD(data.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return {
            'macd': pd.Series(macd_line, index=data.index),
            'signal': pd.Series(signal_line, index=data.index),
            'histogram': pd.Series(histogram, index=data.index)
        }

    @staticmethod
    def stochastic_rsi(data: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k_smooth: int = 3, d_smooth: int = 3) -> Dict[str, pd.Series]:
        data = ensure_float64(data)
        # 1. 先算 RSI
        rsi_series = talib.RSI(data.values, timeperiod=rsi_period)

        # 2. 对 RSI 算 STOCH (注意：Stoch 需要 High/Low/Close，这里全传 RSI 即可)
        # TA-Lib 的 STOCH 函数支持 slowk_period (即 kSmooth) 和 slowd_period (即 dSmooth)
        slowk, slowd = talib.STOCH(
            rsi_series,
            rsi_series,
            rsi_series,
            fastk_period=stoch_period,
            slowk_period=k_smooth,
            slowk_matype=0,
            slowd_period=d_smooth,
            slowd_matype=0
        )

        return {
            'k_percent': pd.Series(slowk, index=data.index),
            'd_percent': pd.Series(slowd, index=data.index)
        }

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        high = ensure_float64(high)
        low = ensure_float64(low)
        close = ensure_float64(close)
        return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        high = ensure_float64(high)
        low = ensure_float64(low)
        close = ensure_float64(close)
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        close = ensure_float64(close)
        volume = ensure_float64(volume)
        if volume.isna().any():
            volume_filled = volume.ffill().fillna(0)
            return pd.Series(talib.OBV(close.values, volume_filled.values), index=close.index)
        return pd.Series(talib.OBV(close.values, volume.values), index=close.index)

    @staticmethod
    def volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
        volume = ensure_float64(volume)
        if volume.isna().any():
            volume = volume.ffill().fillna(0)
        return pd.Series(talib.SMA(volume.values, timeperiod=period), index=volume.index)

    @staticmethod
    def vwap(df: pd.DataFrame, period: str = None, session_hours: int = None) -> pd.Series:
        """
        [修复版] 计算VWAP - 使用向量化 cumsum 替代 apply，修复多列赋值错误
        :param df: 包含 high, low, close, volume, timestamp 的 DataFrame
        :param period: 'day' (自然日重置), 'session' (固定时长重置), 或 None    (全量累计)
        :param session_hours: 当 period='session' 时的重置周期（小时）
        """
        # 1. 准备基础数据
        if df['volume'].isna().any():
            volume_filled = df['volume'].ffill().fillna(0)
        else:
            volume_filled = df['volume']

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        # 计算成交额 (Volume * Price)
        vp = volume_filled * typical_price

        # 2. 确定分组锚点 (Anchor)
        if period == 'day':
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                ts = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                ts = df['timestamp']
            anchor = ts.dt.date
        elif period == 'session' and session_hours:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                ts = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                ts = df['timestamp']
            # 使用 floor 进行时间对齐 (例如 4h, 12h)
            anchor = ts.dt.floor(f'{session_hours}h')
        else:
            # 模式3: 全量累计 (无分组)
            return vp.cumsum() / volume_filled.cumsum()

        # 3. 向量化分组计算 (Vectorized Calculation)
        # 关键修复: 使用 groupby().cumsum() 保证返回的一定是 Series，且索引与原表对齐
        # 相比 apply()，这避免了 Pandas 尝试重构 DataFrame 的行为
        grouped_vp_cumsum = vp.groupby(anchor).cumsum()
        grouped_vol_cumsum = volume_filled.groupby(anchor).cumsum()

        return grouped_vp_cumsum / grouped_vol_cumsum

    @staticmethod
    def fibonacci_retracement(df: pd.DataFrame, period: int = 100) -> Dict[str, float]:
        recent_df = df.tail(period)
        swing_high = recent_df['high'].max()
        swing_low = recent_df['low'].min()
        if pd.isna(swing_high) or pd.isna(swing_low):
            return {f'fib_{int(r*1000):04d}': None for r in [0.236, 0.382, 0.5, 0.618, 0.786]}
        diff = swing_high - swing_low
        ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        levels = {f'fib_{int(r*1000):04d}': round_float(float(swing_high - diff * r)) for r in ratios}
        return levels

    @staticmethod
    def candlestick_patterns(df: pd.DataFrame) -> Dict[str, pd.Series]:
        try:
            open_data = df['open'].astype('float64')
            high_data = df['high'].astype('float64')
            low_data = df['low'].astype('float64')
            close_data = df['close'].astype('float64')

            patterns = {}
            pattern_functions = [
                'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
                'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
                'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
                'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
                'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
                'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
                'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
                'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
                'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
                'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
                'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
                'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
                'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
                'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
                'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
                'CDLXSIDEGAP3METHODS'
            ]

            for pattern_name in pattern_functions:
                try:
                    pattern_func = getattr(talib, pattern_name)
                    result = pattern_func(open_data.values, high_data.values, low_data.values, close_data.values)
                    patterns[pattern_name] = pd.Series(result, index=df.index)
                except Exception as e:
                    patterns[pattern_name] = pd.Series(0, index=df.index)

            return patterns
        except Exception as e:
            logger.error(f"蜡烛形态识别失败: {e}")
            raise

    @staticmethod
    def turtle_trading(df: pd.DataFrame, entry_period: int, exit_period: int, atr_period: int) -> Dict[str, pd.Series]:
        try:
            upper_channel = df['high'].rolling(window=entry_period).max()
            lower_channel = df['low'].rolling(window=entry_period).min()
            exit_upper = df['high'].rolling(window=exit_period).max()
            exit_lower = df['low'].rolling(window=exit_period).min()
            atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], atr_period)

            entry_signal = pd.Series(0, index=df.index)
            entry_signal[df['close'] > upper_channel.shift(1)] = 1
            entry_signal[df['close'] < lower_channel.shift(1)] = -1

            exit_signal = pd.Series(0, index=df.index)
            exit_signal[df['close'] < exit_lower.shift(1)] = -1
            exit_signal[df['close'] > exit_upper.shift(1)] = 1

            return {
                'upper_channel': upper_channel,
                'lower_channel': lower_channel,
                'exit_upper': exit_upper,
                'exit_lower': exit_lower,
                'atr': atr,
                'entry_signal': entry_signal,
                'exit_signal': exit_signal
            }
        except Exception as e:
            logger.error(f"海龟交易法则计算失败: {e}")
            raise

def clean_nan_values(data):
    if isinstance(data, list):
        return [clean_nan_values(x) for x in data]
    elif isinstance(data, dict):
        return {k: clean_nan_values(v) for k, v in data.items()}
    else:
        return round_float(data) if isinstance(data, (float, int)) else data

def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    try:
        if not (df['high'] >= df['low']).all():
            return False
        if not ((df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
            return False
        if (df['volume'] < 0).any():
            return False
        return True
    except Exception as e:
        logger.error(f"数据验证失败: {e}")
        return False

def calculate_all_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    indicators = TechnicalIndicators()
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if df[required_columns].isna().any().any():
        raise ValueError("K线数据包含NaN值")

    for col in required_columns:
        df[col] = ensure_float64(df[col])

    if not validate_ohlcv_data(df):
        raise ValueError("OHLCV数据验证失败")

    results = {}
    results['rsi'] = indicators.rsi(df['close'], config['rsi'])
    results['sma'] = indicators.sma(df['close'], config['sma'])
    results['ema'] = indicators.ema(df['close'], config['ema'])
    results['bollinger_bands'] = indicators.bollinger_bands(df['close'], config['bb'][0], config['bb'][1])
    results['macd'] = indicators.macd(df['close'], config['macd'][0], config['macd'][1], config['macd'][2])
    results['stochastic_rsi'] = indicators.stochastic_rsi(df['close'], rsi_period=config['rsi'], stoch_period=config['stoch_rsi']['period'], k_smooth=config['stoch_rsi']['kSmooth'], d_smooth=config['stoch_rsi']['dSmooth'])
    results['adx'] = indicators.adx(df['high'], df['low'], df['close'], config['adx'])

    if 'atr' in config:
        results['atr'] = indicators.atr(df['high'], df['low'], df['close'], config['atr']['period'])

    if 'obv' in config:
        results['obv'] = indicators.obv(df['close'], df['volume'])

    if 'vwap' in config:
        # 修正: 支持 session_hours 参数
        vwap_cfg = config['vwap']
        results['vwap'] = indicators.vwap(
            df,
            period=vwap_cfg.get('period'),
            session_hours=vwap_cfg.get('session_hours')
        )

    if 'volume_ma' in config:
        results['volume_ma'] = indicators.volume_ma(df['volume'], config['volume_ma'])

    if 'fib' in config:
        results['fibonacci_retracement'] = indicators.fibonacci_retracement(df, config['fib']['period'])

    if config.get('patterns'):
        try:
            results['patterns'] = indicators.candlestick_patterns(df)
        except Exception as e:
            results['patterns'] = {}

    if 'turtle' in config:
        turtle_config = config['turtle']
        min_required = max(turtle_config['entryPeriod'], turtle_config['exitPeriod'], turtle_config['atrPeriod'])
        if len(df) <= min_required:
            results['turtle'] = None
        else:
            try:
                results['turtle'] = indicators.turtle_trading(df, turtle_config['entryPeriod'], turtle_config['exitPeriod'], turtle_config['atrPeriod'])
            except Exception as e:
                results['turtle'] = None

    return results

class BinanceClient:
    """币安API客户端 (增强版：支持分页获取OI)"""

    def __init__(self):
        self.api_url = _config["binance_api_url"]
        self.session = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=_config["request_timeout"])
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """异步获取K线数据 (保持你原有的过滤逻辑)"""
        url = f"{self.api_url}/fapi/v1/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        for attempt in range(_config["retry_attempts"]):
            try:
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    # 转换数值
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

                    df['timestamp_ms'] = df['timestamp'].astype('int64')

                    # --- 你的核心过滤逻辑 ---
                    if not df.empty:
                        last_close_time = int(df.iloc[-1]['close_time'])
                        current_time_ms = int(time.time() * 1000)
                        # 如果 K 线收盘时间还没到，说明是未完成的，移除
                        if last_close_time >= current_time_ms:
                            df = df.iloc[:-1]

                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
            except Exception as e:
                if attempt < _config["retry_attempts"] - 1:
                    await asyncio.sleep(_config["retry_delay"])
                else:
                    raise Exception(f"获取K线数据失败: {e}")

    async def get_open_interest_hist(self, symbol: str, period: str, limit: int = 500) -> pd.DataFrame:
        """
        获取合约持仓量历史 (支持分页)
        
        根据 Binance 文档:
        - limit: default 30, max 500
        - 仅支持最近1个月的数据
        - 若无 startTime 和 endTime，默认返回当前时间往前的limit值
        - IP限频: 1000次/5min
        """
        valid_periods = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]
        if period not in valid_periods:
            return pd.DataFrame()

        # 使用正确的合约数据统计接口 URL
        url = f"{self.api_url}/futures/data/openInterestHist"

        all_data = []
        current_end_time = None
        # Binance 仅支持最近1个月的数据（约30天 = 2592000000 毫秒）
        one_month_ms = 30 * 24 * 60 * 60 * 1000
        earliest_allowed_time = int(time.time() * 1000) - one_month_ms

        # 保护机制：防止死循环，最多请求 10 次
        max_loops = (limit // 500) + 2
        
        # 尝试不同的limit值，从500开始，如果失败则降低
        request_limits = [500, 300, 200, 100, 50]
        request_limit = request_limits[0]

        for loop_count in range(max_loops):
            if len(all_data) >= limit:
                break

            params = {
                'symbol': symbol,
                'period': period,
                'limit': request_limit
            }
            # 如果设置了 endTime，确保不超过1个月限制
            if current_end_time:
                # 如果 endTime 已经超过1个月限制，停止分页
                if current_end_time < earliest_allowed_time:
                    logger.warning(f"OI数据请求已超过1个月限制，停止分页。当前endTime: {current_end_time}, 最早允许: {earliest_allowed_time}")
                    break
                params['endTime'] = current_end_time

            try:
                async with self.session.get(url, params=params) as response:
                    # 先检查状态码，获取详细错误信息
                    if response.status != 200:
                        error_text = await response.text()
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get('msg', error_text)
                        except:
                            error_msg = error_text
                        
                        # 如果是400错误且limit太大，尝试更小的limit
                        if response.status == 400 and request_limit > 50:
                            # 尝试下一个更小的limit
                            limit_index = request_limits.index(request_limit)
                            if limit_index < len(request_limits) - 1:
                                request_limit = request_limits[limit_index + 1]
                                logger.warning(f"获取OI失败，尝试更小的limit: {request_limit}, 错误: {error_msg}")
                                continue
                        
                        logger.warning(f"获取OI分页失败: {response.status}, message='{error_msg}', url='{url}?{urlencode(params)}'")
                        # 如果已经有数据，返回已有数据；否则返回空
                        break
                    
                    data = await response.json()
                    
                    # 检查返回的数据格式
                    if not isinstance(data, list):
                        logger.warning(f"获取OI返回数据格式异常: {type(data)}")
                        break

                    if not data:
                        break

                    # 数据是按时间倒序返回的（最新的在前），所以需要插入到前面
                    all_data = data + all_data

                    # 检查数据是否有timestamp字段
                    if not data or 'timestamp' not in data[0]:
                        logger.warning(f"获取OI数据缺少timestamp字段")
                        break
                    
                    # 获取最早的时间戳（数据是倒序的，最后一个是最早的）
                    last_timestamp = data[-1]['timestamp']
                    
                    # 如果最早的时间戳已经超过1个月限制，停止分页
                    if last_timestamp < earliest_allowed_time:
                        logger.info(f"OI数据已到达1个月限制，停止分页")
                        break
                    
                    # 使用最早的时间戳减1作为下次请求的endTime
                    current_end_time = last_timestamp - 1

            except Exception as e:
                logger.warning(f"获取OI分页失败: {e}")
                # 如果已经有数据，继续处理；否则返回空
                if not all_data:
                    break
                # 如果已经有部分数据，停止分页但返回已有数据
                break

        if not all_data:
            return pd.DataFrame()

        try:
            df = pd.DataFrame(all_data)
            if df.empty:
                return pd.DataFrame()
            
            # 去重并排序（按时间正序）
            df.drop_duplicates(subset=['timestamp'], inplace=True)
            df.sort_values('timestamp', inplace=True)
            
            # 取最后limit条（最新的数据）
            df = df.iloc[-limit:] if len(df) > limit else df

            df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
            df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            return df[['timestamp', 'sumOpenInterestValue']]
        except Exception as e:
            logger.warning(f"处理OI数据失败: {e}")
            return pd.DataFrame()

    async def get_funding_rate_history(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """获取资金费率历史 (无需复杂分页，100条覆盖很久)"""
        url = f"{self.api_url}/fapi/v1/fundingRate"
        params = {'symbol': symbol, 'limit': limit}
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                df = pd.DataFrame(data)
                if df.empty: return pd.DataFrame()

                df['fundingRate'] = df['fundingRate'].astype(float)
                df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
                return df[['fundingTime', 'fundingRate']]
        except Exception:
            return pd.DataFrame()

async def calculate_indicators(symbol: str, interval: str, config: Dict) -> Dict[str, Any]:
    """
    计算技术指标 - 主接口函数 (修复 OI 容差问题 + 安全Token优化)
    """
    try:
        validate_config(config)

        # 1. 特殊处理 1m 周期 (币安无 1m OI，降级取 5m)
        oi_interval = interval
        if interval == "1m":
            oi_interval = "5m"

        async with BinanceClient() as client:
            tasks = [
                client.get_klines(symbol, interval, config['limit']),
                client.get_open_interest_hist(symbol, oi_interval, config['limit']),
                client.get_funding_rate_history(symbol)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            df = results[0]
            if isinstance(df, Exception) or df.empty:
                raise ValueError(f"无法获取K线: {df}")

            oi_df = results[1] if not isinstance(results[1], Exception) else pd.DataFrame()
            fr_df = results[2] if not isinstance(results[2], Exception) else pd.DataFrame()

        # 计算常规指标
        indicators_data = calculate_all_indicators(df, config)

        # 构建基础 DataFrame
        result_df = pd.DataFrame({
            't': df['timestamp_ms'],
            'c': df['close'],
            'v': df['volume'],
            'rsi': indicators_data['rsi'],
            'sma': indicators_data['sma'],
            'ema': indicators_data['ema'],
            'bb_u': indicators_data['bollinger_bands']['upper'],
            'bb_m': indicators_data['bollinger_bands']['middle'],
            'bb_l': indicators_data['bollinger_bands']['lower'],
            'macd': indicators_data['macd']['macd'],
            'macd_s': indicators_data['macd']['signal'],
            'macd_h': indicators_data['macd']['histogram'],
            'k': indicators_data['stochastic_rsi']['k_percent'],
            'd': indicators_data['stochastic_rsi']['d_percent'],
            'adx': indicators_data['adx']
        })

        if 'atr' in indicators_data:
            result_df['atr'] = indicators_data['atr']

        # --- [关键修复1] 计算K线的Close Time用于OI数据对齐 ---
        # 问题：币安K线的timestamp是Open Time，但OI数据应该对齐到收盘时刻
        # 解决：计算 Close Time = Open Time + Interval
        result_df['temp_ts'] = pd.to_datetime(result_df['t'], unit='ms')
        
        try:
            interval_delta = interval_to_timedelta(interval)
            result_df['close_ts'] = result_df['temp_ts'] + interval_delta
            logger.debug(f"✅ 计算Close Time成功: interval={interval}, delta={interval_delta}")
        except ValueError as e:
            logger.error(f"❌ 时间间隔转换失败: {e}")
            raise

        if not oi_df.empty:
            oi_df.set_index('timestamp', inplace=True)
            oi_df.sort_index(inplace=True)

            # 根据 K 线周期动态设置容差
            if interval == "1m":
                tolerance = pd.Timedelta('10m')
            elif interval in ["5m", "15m", "30m"]:
                tolerance = pd.Timedelta('1h')
            elif interval in ["1h", "2h", "4h", "6h", "8h", "12h"]:
                tolerance = pd.Timedelta('12h') # 增加容差以确保捕获大周期数据
            else:
                tolerance = pd.Timedelta('24h')

            result_df = pd.merge_asof(
                result_df,
                oi_df[['sumOpenInterestValue']],
                left_on='close_ts',  # [修复] 使用Close Time而非Open Time
                right_index=True,
                direction='backward',
                tolerance=tolerance
            )
            result_df.rename(columns={'sumOpenInterestValue': 'oi'}, inplace=True)
            result_df['oi'] = result_df['oi'].ffill().fillna(0)
        else:
            result_df['oi'] = 0

        # Funding Rate 合并
        if not fr_df.empty:
            fr_df.set_index('fundingTime', inplace=True)
            fr_df.sort_index(inplace=True)
            result_df = pd.merge_asof(
                result_df,
                fr_df[['fundingRate']],
                left_on='close_ts',  # [修复] 统一使用Close Time
                right_index=True,
                direction='backward',
                tolerance=pd.Timedelta('24h')
            )
            result_df.rename(columns={'fundingRate': 'funding'}, inplace=True)
            result_df['funding'] = result_df['funding'].ffill().fillna(0)
        else:
            result_df['funding'] = 0

        # 清理临时列（保留close_ts用于后续验证）
        if 'temp_ts' in result_df.columns:
            result_df.drop(columns=['temp_ts'], inplace=True)

        # --- 后续逻辑保持不变 ---
        if 'obv' in indicators_data: result_df['obv'] = indicators_data['obv']
        if 'vwap' in indicators_data: result_df['vwap'] = indicators_data['vwap']
        if 'volume_ma' in indicators_data: result_df['v_ma'] = indicators_data['volume_ma']

        if 'patterns' in indicators_data and indicators_data['patterns']:
            pattern_series_dict = indicators_data['patterns']
            patterns_list = []
            for i in range(len(df)):
                active_patterns = []
                for name, series in pattern_series_dict.items():
                    val = series.iloc[i]
                    if val != 0:
                        active_patterns.append(f"{name}:{int(val)}")
                patterns_list.append(active_patterns if active_patterns else None)
            result_df['ptn'] = patterns_list

        if 'turtle' in indicators_data and indicators_data['turtle']:
            td = indicators_data['turtle']
            result_df['t_up'] = td['upper_channel']
            result_df['t_dn'] = td['lower_channel']
            result_df['t_sig'] = td['entry_signal']

        key_indicators = ['rsi', 'sma', 'ema', 'macd', 'adx']
        filter_cols = [k for k in key_indicators if k in result_df.columns]
        if filter_cols:
            result_df = result_df.dropna(subset=filter_cols)

        final_data = {}
        for col in result_df.columns:
            series = result_df[col]
            if col == 'ptn':
                final_data[col] = series.tolist()
            else:
                # 默认使用高精度
                final_data[col] = [round_float(x) for x in series.tolist()]

        keys_to_remove = [k for k, v in final_data.items() if all(x is None for x in v)]
        for k in keys_to_remove:
            del final_data[k]

        # --- [关键修复2] 数据完整性验证 ---
        expected_length = len(result_df)
        try:
            validate_output_data_alignment(final_data, expected_length)
            logger.debug(f"✅ 数据对齐验证通过: {len(final_data)}列, 每列{expected_length}行")
        except ValueError as e:
            logger.error(f"❌ 数据对齐验证失败: {e}")
            raise

        # --- [关键修复3] 计算OI数据有效性信息 ---
        oi_data_info = None
        if 'oi' in final_data and 'oi' in result_df.columns:
            oi_series = result_df['oi']
            oi_data_info = calculate_oi_data_info(oi_series, expected_length)
            
            # 如果覆盖率 < 50%，记录警告
            if oi_data_info['oi_coverage_percent'] < 50:
                logger.warning(
                    f"⚠️ OI数据覆盖率较低: {oi_data_info['oi_coverage_percent']:.1f}% "
                    f"({oi_data_info['oi_valid_bars']}/{oi_data_info['total_bars']}条有效) "
                    f"- {symbol} {interval}"
                )
            else:
                logger.debug(
                    f"✅ OI数据覆盖率: {oi_data_info['oi_coverage_percent']:.1f}% "
                    f"({oi_data_info['oi_valid_bars']}/{oi_data_info['total_bars']}条有效)"
                )

        # 清理close_ts列（验证完成后）
        if 'close_ts' in result_df.columns:
            result_df.drop(columns=['close_ts'], inplace=True)

        result = {'indicators': final_data}
        
        # 添加OI数据信息（如果存在）
        if oi_data_info:
            result['oi_data_info'] = oi_data_info
        
        if 'fibonacci_retracement' in indicators_data:
            result['fibonacci'] = indicators_data['fibonacci_retracement']

        # --- [优化] 安全Token优化 (仅针对时间戳和量) ---

        # 1. 时间戳转整数 (去除 .0)
        if 't' in final_data:
            final_data['t'] = [int(x) for x in final_data['t']]

        # 2. 量类指标转整数 (去除小数位) - 安全操作，不影响小币种价格
        volume_keys = ['v', 'oi', 'obv']
        for k in volume_keys:
            if k in final_data:
                final_data[k] = [int(x) if x is not None and not pd.isna(x) else None for x in final_data[k]]

        return result

    except Exception as e:
        logger.error(f"计算失败: {e}")
        return {'error': str(e), 'symbol': symbol, 'interval': interval}

def validate_config(config: Dict) -> None:
    if not config: raise ValueError("config参数不能为空")
    required_keys = ['limit', 'rsi', 'macd', 'bb', 'sma', 'ema', 'adx', 'stoch_rsi']
    for key in required_keys:
        if key not in config: raise ValueError(f"config中缺少必需的参数: {key}")
    if not isinstance(config['stoch_rsi'], dict): raise ValueError("stoch_rsi必须是字典类型")


def calculate_indicators_sync(symbol: str, interval: str, config: Dict) -> Dict[str, Any]:
    try:
        validate_config(config)
        future = _thread_pool.submit(_calculate_indicators_worker, symbol, interval, config)
        return future.result(timeout=_config["request_timeout"])
    except Exception as e:
        logger.error(f"计算技术指标失败: {e}")
        return {'error': str(e), 'symbol': symbol, 'interval': interval}

def _calculate_indicators_worker(symbol: str, interval: str, config: Dict) -> Dict[str, Any]:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(calculate_indicators(symbol, interval, config))
        return result
    except Exception as e:
        logger.error(f"线程池计算失败: {e}")
        return {'error': str(e), 'symbol': symbol, 'interval': interval}
    finally:
        loop.close()

def get_supported_intervals() -> List[str]:
    """获取支持的时间间隔列表 (已移除 3m)"""
    return ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]

def shutdown_thread_pool():
    global _thread_pool
    if _thread_pool:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None