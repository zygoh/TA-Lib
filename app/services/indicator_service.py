
"""
币安技术指标计算服务 - 使用TA-Lib
包含API调用和指标计算功能，支持按日重置 VWAP 和过滤 null 数据
"""
import asyncio
import aiohttp
import pandas as pd
import talib
from typing import Dict, Any, List
import logging
import json
import os
from concurrent.futures import ThreadPoolExecutor
from aiohttp.client_exceptions import ClientResponseError
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置文件
def load_config():
    """加载配置文件"""
    # 配置文件在项目根目录的config文件夹中
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

class TechnicalIndicators:
    """技术指标计算类 - 使用TA-Lib库"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """简单移动平均线"""
        data = ensure_float64(data)
        return pd.Series(talib.SMA(data.values, timeperiod=period), index=data.index)
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """指数移动平均线"""
        data = ensure_float64(data)
        return pd.Series(talib.EMA(data.values, timeperiod=period), index=data.index)
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指数"""
        data = ensure_float64(data)
        return pd.Series(talib.RSI(data.values, timeperiod=period), index=data.index)
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """布林带"""
        data = ensure_float64(data)
        upper, middle, lower = talib.BBANDS(data.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
        return {
            'upper': pd.Series(upper, index=data.index),
            'middle': pd.Series(middle, index=data.index),
            'lower': pd.Series(lower, index=data.index)
        }
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD指标"""
        data = ensure_float64(data)
        macd_line, signal_line, histogram = talib.MACD(data.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return {
            'macd': pd.Series(macd_line, index=data.index),
            'signal': pd.Series(signal_line, index=data.index),
            'histogram': pd.Series(histogram, index=data.index)
        }
    
    @staticmethod
    def stochastic_rsi(data: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k_smooth: int = 3, d_smooth: int = 3) -> Dict[str, pd.Series]:
        """随机RSI"""
        data = ensure_float64(data)
        k_percent, d_percent = talib.STOCHRSI(data.values, timeperiod=rsi_period, fastk_period=stoch_period, fastd_period=d_smooth, fastd_matype=0)
        return {
            'k_percent': pd.Series(k_percent, index=data.index),
            'd_percent': pd.Series(d_percent, index=data.index)
        }
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均趋向指数"""
        high = ensure_float64(high)
        low = ensure_float64(low)
        close = ensure_float64(close)
        return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均真实波幅 (ATR)"""
        high = ensure_float64(high)
        low = ensure_float64(low)
        close = ensure_float64(close)
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """成交量净额 (OBV)"""
        close = ensure_float64(close)
        volume = ensure_float64(volume)
        if volume.isna().any():
            logger.warning("OBV 计算失败：volume 包含 NaN，使用前向填充")
            volume_filled = volume.ffill().fillna(0)
            return pd.Series(talib.OBV(close.values, volume_filled.values), index=close.index)
        return pd.Series(talib.OBV(close.values, volume.values), index=close.index)
    
    @staticmethod
    def vwap(df: pd.DataFrame, period: str = None) -> pd.Series:
        """成交量加权平均价 (VWAP)，支持按日重置"""
        if df['volume'].isna().any():
            logger.warning("VWAP 计算失败：volume 包含 NaN，使用前向填充")
            volume_filled = df['volume'].ffill().fillna(0)
        else:
            volume_filled = df['volume']
            
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        if period == 'day':
            # 动态按日分组，不依赖硬编码的K线数量
            df_copy = df.copy()
            # timestamp 已经是 datetime 类型，直接提取日期
            if pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                df_copy['date'] = df_copy['timestamp'].dt.date
            else:
                df_copy['date'] = pd.to_datetime(df_copy['timestamp'], unit='ms').dt.date
            
            def calc_vwap(x):
                # x 是每个分组的 DataFrame，但我们只需要索引
                # 通过索引访问外部的 volume_filled 和 typical_price
                vol = volume_filled.loc[x.index]
                tp = typical_price.loc[x.index]
                return (vol * tp).cumsum() / vol.cumsum()
            
            # 使用除了 'date' 之外的列进行 groupby，避免分组列被包含在 apply 中
            vwap = df_copy.drop(columns=['date']).groupby(df_copy['date'], group_keys=False).apply(calc_vwap)
            return vwap
        else:
            # 默认累积 VWAP
            vwap = (typical_price * volume_filled).cumsum() / volume_filled.cumsum()
            return vwap
    
    @staticmethod
    def fibonacci_retracement(df: pd.DataFrame, period: int = 100) -> Dict[str, float]:
        """斐波那契回撤位 (基于最近 period 根 K 线的 swing high/low)"""
        recent_df = df.tail(period)
        swing_high = recent_df['high'].max()
        swing_low = recent_df['low'].min()
        if pd.isna(swing_high) or pd.isna(swing_low):
            logger.warning("斐波那契计算失败：high 或 low 包含 NaN")
            return {f'fib_{int(r*1000):04d}': None for r in [0.236, 0.382, 0.5, 0.618, 0.786]}
        diff = swing_high - swing_low
        ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        levels = {f'fib_{int(r*1000):04d}': float(swing_high - diff * r) for r in ratios}
        return levels
    
    @staticmethod
    def candlestick_patterns(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        识别蜡烛图形态
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
            
        Returns:
            dict: 每个形态的识别结果 Series (100=看涨, -100=看跌, 0=无)
        """
        try:
            # 确保数据类型为 float64（TA-Lib 要求）
            open_data = df['open'].astype('float64')
            high_data = df['high'].astype('float64')
            low_data = df['low'].astype('float64')
            close_data = df['close'].astype('float64')
            
            patterns = {}
            
            # 定义所有支持的蜡烛形态函数
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
            
            # 调用每个形态识别函数
            for pattern_name in pattern_functions:
                try:
                    pattern_func = getattr(talib, pattern_name)
                    result = pattern_func(open_data.values, high_data.values, low_data.values, close_data.values)
                    patterns[pattern_name] = pd.Series(result, index=df.index)
                except Exception as e:
                    logger.warning(f"蜡烛形态 {pattern_name} 计算失败: {e}")
                    patterns[pattern_name] = pd.Series(0, index=df.index)
            
            logger.info(f"成功识别 {len(patterns)} 种蜡烛形态")
            return patterns
            
        except Exception as e:
            logger.error(f"蜡烛形态识别失败: {e}")
            raise
    
    @staticmethod
    def turtle_trading(df: pd.DataFrame, entry_period: int, exit_period: int, atr_period: int) -> Dict[str, pd.Series]:
        """
        计算海龟交易法则指标
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
            entry_period: 入场周期（唐奇安通道周期）
            exit_period: 出场周期
            atr_period: ATR 计算周期
            
        Returns:
            dict: 包含通道值、信号和 ATR 的字典
            {
                'upper_channel': pd.Series,  # 入场上轨
                'lower_channel': pd.Series,  # 入场下轨
                'exit_upper': pd.Series,     # 出场上轨
                'exit_lower': pd.Series,     # 出场下轨
                'atr': pd.Series,            # ATR 值
                'entry_signal': pd.Series,   # 入场信号 (1=多头, -1=空头, 0=无)
                'exit_signal': pd.Series     # 出场信号 (1=空头平仓, -1=多头平仓, 0=无)
            }
        """
        try:
            # 计算唐奇安通道（入场）
            upper_channel = df['high'].rolling(window=entry_period).max()
            lower_channel = df['low'].rolling(window=entry_period).min()
            
            # 计算唐奇安通道（出场）
            exit_upper = df['high'].rolling(window=exit_period).max()
            exit_lower = df['low'].rolling(window=exit_period).min()
            
            # 计算 ATR
            atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], atr_period)
            
            # 计算入场信号
            entry_signal = pd.Series(0, index=df.index)
            entry_signal[df['close'] > upper_channel.shift(1)] = 1   # 多头入场
            entry_signal[df['close'] < lower_channel.shift(1)] = -1  # 空头入场
            
            # 计算出场信号
            exit_signal = pd.Series(0, index=df.index)
            exit_signal[df['close'] < exit_lower.shift(1)] = -1  # 多头平仓
            exit_signal[df['close'] > exit_upper.shift(1)] = 1   # 空头平仓
            
            logger.info(f"海龟交易法则计算完成: entry_period={entry_period}, exit_period={exit_period}, atr_period={atr_period}")
            
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
    

def ensure_float64(data: pd.Series) -> pd.Series:
    """确保数据为float64类型，TA-Lib要求"""
    if data.dtype != 'float64':
        logger.debug(f"转换数据类型从 {data.dtype} 到 float64")
        return data.astype('float64')
    return data

def to_list_preserve_precision(series: pd.Series) -> List:
    """转换为列表，保持原始精度"""
    return [None if pd.isna(x) else float(x) for x in series]

def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    try:
        # 检查价格逻辑
        if not (df['high'] >= df['low']).all():
            logger.error("数据验证失败：high < low")
            return False
        if not ((df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
            logger.error("数据验证失败：high < open 或 high < close")
            return False
        if not ((df['low'] <= df['open']) & (df['low'] <= df['close'])).all():
            logger.error("数据验证失败：low > open 或 low > close")
            return False
        
        # 检查成交量
        if (df['volume'] < 0).any():
            logger.error("数据验证失败：volume < 0")
            return False
        
        # 检查价格变化幅度（防止异常数据）
        price_changes = df['close'].pct_change().abs()
        if (price_changes > 0.5).any():  # 单根K线涨跌幅超过50%
            logger.warning("检测到异常价格变化，可能存在数据问题")
        
        return True
    except Exception as e:
        logger.error(f"数据验证失败: {e}")
        return False

def calculate_all_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算所有技术指标
    
    Args:
        df: 包含OHLCV数据的DataFrame
        config: 技术指标配置参数
        
    Returns:
        dict: 包含所有技术指标的字典
    """
    indicators = TechnicalIndicators()
    
    # 验证输入数据
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if df[required_columns].isna().any().any():
        logger.warning(f"K线数据包含NaN值: {df[required_columns].isna().sum().to_dict()}")
        raise ValueError("K线数据包含NaN值")
    
    # 确保数据类型为float64（TA-Lib要求）
    for col in required_columns:
        df[col] = ensure_float64(df[col])
        logger.debug(f"列 {col} 数据类型: {df[col].dtype}")
    
    # 验证OHLCV数据合理性
    if not validate_ohlcv_data(df):
        raise ValueError("OHLCV数据验证失败")
    
    # 计算最小所需 K 线数量
    min_data_required = max(
        config['rsi'] + 1,  # RSI
        config['sma'],      # SMA
        config['ema'],      # EMA
        config['bb'][0],    # Bollinger Bands
        config['macd'][1] + config['macd'][2],  # MACD
        config['rsi'] + config['stoch_rsi']['period'] + max(config['stoch_rsi']['kSmooth'], config['stoch_rsi']['dSmooth']) - 1,  # Stochastic RSI
        config['adx'] + 1,  # ADX
        config.get('atr', {}).get('period', 14) + 1 if 'atr' in config else 0,  # ATR
        # 注意：斐波那契回撤不需要额外的数据，它只是基于现有数据计算回撤位
        max(config['turtle']['entryPeriod'], config['turtle']['exitPeriod'], config['turtle']['atrPeriod']) + 1 if 'turtle' in config else 0  # Turtle Trading
    )
    logger.info(f"最小所需 K 线数量: {min_data_required}, 实际: {len(df)}")
    if len(df) < min_data_required:
        raise ValueError(f"数据不足，需要至少 {min_data_required} 根K线，实际 {len(df)} 根")
    
    results = {}
    
    # RSI
    results['rsi'] = to_list_preserve_precision(indicators.rsi(df['close'], config['rsi']))
    
    # SMA
    results['sma'] = to_list_preserve_precision(indicators.sma(df['close'], config['sma']))
    
    # EMA
    results['ema'] = to_list_preserve_precision(indicators.ema(df['close'], config['ema']))
    
    # 布林带
    bb = indicators.bollinger_bands(df['close'], config['bb'][0], config['bb'][1])
    results['bollinger_bands'] = {
        'upper': to_list_preserve_precision(bb['upper']),
        'middle': to_list_preserve_precision(bb['middle']),
        'lower': to_list_preserve_precision(bb['lower'])
    }
    
    # MACD
    macd = indicators.macd(df['close'], config['macd'][0], config['macd'][1], config['macd'][2])
    results['macd'] = {
        'macd': to_list_preserve_precision(macd['macd']),
        'signal': to_list_preserve_precision(macd['signal']),
        'histogram': to_list_preserve_precision(macd['histogram'])
    }
    
    # 随机RSI
    stoch_rsi = indicators.stochastic_rsi(
        df['close'], 
        rsi_period=config['rsi'],
        stoch_period=config['stoch_rsi']['period'],
        k_smooth=config['stoch_rsi']['kSmooth'],
        d_smooth=config['stoch_rsi']['dSmooth']
    )
    results['stochastic_rsi'] = {
        'k_percent': to_list_preserve_precision(stoch_rsi['k_percent']),
        'd_percent': to_list_preserve_precision(stoch_rsi['d_percent'])
    }
    
    # ADX
    results['adx'] = to_list_preserve_precision(indicators.adx(df['high'], df['low'], df['close'], config['adx']))
    
    # ATR
    if 'atr' in config:
        results['atr'] = to_list_preserve_precision(indicators.atr(df['high'], df['low'], df['close'], config['atr']['period']))
    
    # OBV
    if 'obv' in config:
        results['obv'] = to_list_preserve_precision(indicators.obv(df['close'], df['volume']))
    
    # VWAP
    if 'vwap' in config:
        results['vwap'] = to_list_preserve_precision(indicators.vwap(df, config['vwap'].get('period')))
    
    # Fibonacci Retracement
    if 'fib' in config:
        results['fibonacci_retracement'] = indicators.fibonacci_retracement(df, config['fib']['period'])
    
    # 蜡烛形态识别
    if config.get('patterns'):
        try:
            patterns_result = indicators.candlestick_patterns(df)
            results['patterns'] = patterns_result
        except Exception as e:
            logger.error(f"蜡烛形态识别失败: {e}")
            results['patterns'] = {}
    
    # 海龟交易法则
    if 'turtle' in config:
        turtle_config = config['turtle']
        # 检查数据是否足够
        min_required = max(
            turtle_config['entryPeriod'],
            turtle_config['exitPeriod'],
            turtle_config['atrPeriod']
        )
        
        if len(df) <= min_required:
            logger.warning(f"海龟法则数据不足: 至少需要 {min_required + 1} 根K线，实际 {len(df)} 根")
            results['turtle'] = None
        else:
            try:
                turtle_result = indicators.turtle_trading(
                    df,
                    turtle_config['entryPeriod'],
                    turtle_config['exitPeriod'],
                    turtle_config['atrPeriod']
                )
                results['turtle'] = {
                    'upper_channel': to_list_preserve_precision(turtle_result['upper_channel']),
                    'lower_channel': to_list_preserve_precision(turtle_result['lower_channel']),
                    'exit_upper': to_list_preserve_precision(turtle_result['exit_upper']),
                    'exit_lower': to_list_preserve_precision(turtle_result['exit_lower']),
                    'atr': to_list_preserve_precision(turtle_result['atr']),
                    'entry_signal': to_list_preserve_precision(turtle_result['entry_signal']),
                    'exit_signal': to_list_preserve_precision(turtle_result['exit_signal'])
                }
            except Exception as e:
                logger.error(f"海龟法则计算失败: {e}")
                results['turtle'] = None
    
    # 清理 NaN
    results = clean_nan_values(results)
    
    return results

def clean_nan_values(data):
    """清理数据中的NaN值，转换为None"""
    if isinstance(data, list):
        return [None if pd.isna(x) else x for x in data]
    elif isinstance(data, dict):
        return {k: clean_nan_values(v) for k, v in data.items()}
    elif isinstance(data, pd.Series):
        # 如果是 Series，不处理，直接返回
        return data
    else:
        # 只对标量值检查 NaN
        try:
            if pd.isna(data):
                return None
        except (TypeError, ValueError):
            # 如果 pd.isna 失败，说明不是可以检查 NaN 的类型
            pass
    return data

class BinanceClient:
    """币安API客户端"""
    
    def __init__(self):
        self.api_url = _config["binance_api_url"]
        self.session = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=_config["request_timeout"])
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=True
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """异步获取K线数据，支持重试"""
        url = f"{self.api_url}/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        logger.info(f"请求URL: {url}, 参数: {params}")
        for attempt in range(_config["retry_attempts"]):
            try:
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    logger.info(f"获取K线数据: {symbol} {interval}, 数据点数: {len(data)}")
                    
                    # 验证返回数据量
                    if len(data) != limit:
                        logger.warning(f"预期 {limit} 根K线，实际返回 {len(data)} 根")
                    
                    # 转换为DataFrame
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    # 转换数据类型为float64（TA-Lib要求）
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                    
                    # 保存原始时间戳（毫秒）
                    df['timestamp_ms'] = df['timestamp'].astype(int)
                    
                    # 转换时间戳为datetime用于排序
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # 验证时间戳连续性（允许5%的容差）
                    interval_ms = {
                        '1m': 60_000, '5m': 300_000, '15m': 900_000, '30m': 1_800_000,
                        '1h': 3_600_000, '2h': 7_200_000, '4h': 14_400_000, '1d': 86_400_000
                    }.get(interval)
                    if interval_ms:
                        time_diffs = df['timestamp_ms'].diff().dropna()
                        tolerance = interval_ms * 0.05  # 5%容差
                        if not ((time_diffs >= interval_ms - tolerance) & (time_diffs <= interval_ms + tolerance)).all():
                            logger.warning(f"K线时间戳不连续: {symbol} {interval}，但继续处理")
                    
                    logger.info(f"数据排序验证: 开始时间 {df['timestamp'].iloc[0]}, 结束时间 {df['timestamp'].iloc[-1]}")
                    return df
                
            except ClientResponseError as e:
                if attempt < _config["retry_attempts"] - 1:
                    logger.warning(f"请求失败，第 {attempt + 1} 次重试: {e}")
                    await asyncio.sleep(_config["retry_delay"])
                else:
                    logger.error(f"获取K线数据失败: {e}")
                    raise Exception(f"获取K线数据失败: {e}")
            except Exception as e:
                logger.error(f"获取K线数据失败: {e}")
                raise Exception(f"获取K线数据失败: {e}")

def validate_config(config: Dict) -> None:
    """验证配置参数"""
    if not config:
        raise ValueError("config参数不能为空")
    
    required_keys = ['limit', 'rsi', 'macd', 'bb', 'sma', 'ema', 'adx', 'stoch_rsi']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"config中缺少必需的参数: {key}")
    
    if not isinstance(config['stoch_rsi'], dict):
        raise ValueError("stoch_rsi必须是字典类型")
    
    stoch_rsi_keys = ['period', 'kSmooth', 'dSmooth']
    for key in stoch_rsi_keys:
        if key not in config['stoch_rsi']:
            raise ValueError(f"stoch_rsi中缺少必需的参数: {key}")
    
    # 验证新增参数
    if 'atr' in config and not isinstance(config['atr'], dict):
        raise ValueError("atr 必须是字典类型")
    if 'obv' in config and not isinstance(config['obv'], bool):
        raise ValueError("obv 必须是布尔类型")
    if 'vwap' in config and not isinstance(config['vwap'], dict):
        raise ValueError("vwap 必须是字典类型")
    if 'fib' in config and not isinstance(config['fib'], dict):
        raise ValueError("fib 必须是字典类型")
    
    # 验证蜡烛形态参数
    if 'patterns' in config:
        if not isinstance(config['patterns'], bool):
            raise ValueError("patterns 必须是布尔类型")
    
    # 验证海龟法则参数
    if 'turtle' in config:
        if not isinstance(config['turtle'], dict):
            raise ValueError("turtle 必须是字典类型")
        
        turtle_keys = ['entryPeriod', 'exitPeriod', 'atrPeriod']
        for key in turtle_keys:
            if key not in config['turtle']:
                raise ValueError(f"turtle 中缺少必需的参数: {key}")
            if not isinstance(config['turtle'][key], int) or config['turtle'][key] <= 0:
                raise ValueError(f"turtle.{key} 必须是正整数")

async def calculate_indicators(symbol: str, interval: str, config: Dict) -> Dict[str, Any]:
    """
    计算技术指标 - 主接口函数，过滤 null 数据
    
    Args:
        symbol: 交易对符号，如 'BTCUSDT'
        interval: 时间间隔，如 '15m', '1h', '1d'
        config: 技术指标配置参数
        
    Returns:
        dict: 包含所有技术指标的字典，仅包含完整指标行
    """
    try:
        # 验证配置参数
        validate_config(config)
        
        # 获取K线数据
        async with BinanceClient() as client:
            df = await client.get_klines(symbol, interval, config['limit'])
        
        if df.empty:
            raise ValueError(f"无法获取 {symbol} 的K线数据")
        
        # 计算技术指标
        indicators_data = calculate_all_indicators(df, config)
        
        # 构建最新数据点（而不是完整时间序列）
        latest_index = -1  # 最后一个数据点
        
        result = {
            'timestamp': int(df['timestamp_ms'].iloc[latest_index]),
            'close': float(df['close'].iloc[latest_index]),
            'rsi': indicators_data['rsi'][latest_index],
            'sma': indicators_data['sma'][latest_index],
            'ema': indicators_data['ema'][latest_index],
            'bb_upper': indicators_data['bollinger_bands']['upper'][latest_index],
            'bb_basis': indicators_data['bollinger_bands']['middle'][latest_index],
            'bb_lower': indicators_data['bollinger_bands']['lower'][latest_index],
            'macd': indicators_data['macd']['macd'][latest_index],
            'macd_signal': indicators_data['macd']['signal'][latest_index],
            'macd_histogram': indicators_data['macd']['histogram'][latest_index],
            'stoch_rsi_k': indicators_data['stochastic_rsi']['k_percent'][latest_index],
            'stoch_rsi_d': indicators_data['stochastic_rsi']['d_percent'][latest_index],
            'adx': indicators_data['adx'][latest_index],
            'atr': indicators_data.get('atr', [None] * len(df))[latest_index],
            'obv': indicators_data.get('obv', [None] * len(df))[latest_index],
            'vwap': indicators_data.get('vwap', [None] * len(df))[latest_index]
        }
        
        # 添加斐波那契回撤位（如果存在）
        if 'fibonacci_retracement' in indicators_data:
            for level, value in indicators_data['fibonacci_retracement'].items():
                result[level] = value
        
        # 添加蜡烛形态（如果启用）
        if 'patterns' in indicators_data:
            # 提取最新K线的所有形态
            latest_patterns = []
            for pattern_name, pattern_values in indicators_data['patterns'].items():
                # 获取最新值，确保是标量
                pattern_value = pattern_values.iloc[latest_index]
                if isinstance(pattern_value, pd.Series):
                    pattern_value = pattern_value.iloc[0]
                
                if pattern_value != 0:
                    latest_patterns.append({
                        'name': pattern_name,
                        'value': int(pattern_value)
                    })
            result['patterns'] = latest_patterns
        
        # 添加海龟法则（如果启用）
        if 'turtle' in indicators_data:
            turtle_data = indicators_data['turtle']
            if turtle_data is not None and isinstance(turtle_data, dict):
                result['turtle'] = {
                    'upper_channel': turtle_data['upper_channel'][latest_index],
                    'lower_channel': turtle_data['lower_channel'][latest_index],
                    'exit_upper': turtle_data['exit_upper'][latest_index],
                    'exit_lower': turtle_data['exit_lower'][latest_index],
                    'atr': turtle_data['atr'][latest_index],
                    'entry_signal': turtle_data['entry_signal'][latest_index],
                    'exit_signal': turtle_data['exit_signal'][latest_index]
                }
        
        logger.info(f"技术指标计算完成: {symbol} {interval}")
        return result
    
    except Exception as e:
        logger.error(f"计算技术指标失败: {e}")
        return {
            'error': str(e),
            'symbol': symbol,
            'interval': interval
        }

def calculate_indicators_sync(symbol: str, interval: str, config: Dict) -> Dict[str, Any]:
    """
    同步计算技术指标 - 使用线程池支持并发
    """
    try:
        validate_config(config)
        future = _thread_pool.submit(_calculate_indicators_worker, symbol, interval, config)
        result = future.result(timeout=_config["request_timeout"])
        return result
    except Exception as e:
        logger.error(f"计算技术指标失败: {e}")
        return {
            'error': str(e),
            'symbol': symbol,
            'interval': interval
        }

def _calculate_indicators_worker(symbol: str, interval: str, config: Dict) -> Dict[str, Any]:
    """
    在线程池中执行的技术指标计算工作函数
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(calculate_indicators(symbol, interval, config))
        return result
    except Exception as e:
        logger.error(f"线程池计算失败: {e}")
        return {
            'error': str(e),
            'symbol': symbol,
            'interval': interval
        }
    finally:
        loop.close()

def get_supported_intervals() -> List[str]:
    """获取支持的时间间隔列表"""
    return ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]


def shutdown_thread_pool():
    """关闭线程池"""
    global _thread_pool
    if _thread_pool:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None