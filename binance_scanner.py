import asyncio
import aiohttp
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import ta
from ta.trend import ema_indicator
from ta.momentum import rsi
from ta.volatility import BollingerBands
from ta.trend import macd
from kivy.clock import Clock
from android import mActivity
from jnius import autoclass

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Binance API Configuration
BASE_URL = "https://fapi.binance.com"
API_ENDPOINT = "/fapi/v1/klines"
TIME_ENDPOINT = "/fapi/v1/time"

# Scanner Configuration
MAX_CONCURRENT_REQUESTS = 50
REQUEST_INTERVAL = 0.02
LOOKBACK_CANDLES = 3
MIN_CANDLES_NEEDED = 60  # Enough for all indicators

# Combination Switch Configuration (will be set by UI)
USE_RSI_KRI = True       # RSI (20/65) + KRI (4/-6) combination
USE_EMA_KRI = True      # EMA + KRI (5/-5) combination  
USE_MACD_KRI = True     # MACD + KRI (5/-5) combination
USE_BB_KRI = True       # Bollinger Bands + KRI (5/-5) combination
USE_CROSSOVER_KRI = True # MACD Crossover + KRI (5/-5) combination

# New EMA Switch Configuration (will be set by UI)
USE_CURRENT_TF_EMA = True  # Switch for current timeframe EMA conditions
USE_HIGHER_TF_CONFIRMATION = True  # Switch for higher timeframe confirmation
CURRENT_TF_EMA9 = 5 #EMA6(9) > EMA15(20) > EMA30(50) and price > EMA6(9) for buy, EMA6 < EMA15 < EMA30 and price < EMA6 for sell
CURRENT_TF_EMA20 = 10
CURRENT_TF_EMA50 = 50
HIGHER_TF_EMA = 50
HIGHER_TF_CANDLES = 60

# Indicator Parameters
RSI_OVERBOUGHT = 70      # RSI sell threshold
RSI_OVERSOLD = 30        # RSI buy threshold
KRI_UPPER_RSI =  3       # KRI upper threshold for RSI combination
KRI_LOWER_RSI = -3       # KRI lower threshold for RSI combination
KRI_UPPER_OTHER = 3      # KRI upper threshold for other combinations
KRI_LOWER_OTHER = -3     # KRI lower threshold for other combinations
KRI_MA_PERIOD = 20       # KRI moving average period
EMA_PERIOD = 20          # EMA period for EMA+KRI combination
BB_PERIOD = 20           # Bollinger Bands period
BB_STDDEV = 2            # Bollinger Bands standard deviations
MACD_FAST = 12           # MACD fast period
MACD_SLOW = 26           # MACD slow period
MACD_SIGNAL = 9          # MACD signal period

# Rate Limiting
KLINES_LIMIT = 1000
RATE_LIMIT_BARRIER = 2400
SOFT_LIMIT = RATE_LIMIT_BARRIER * 0.80
HARD_LIMIT = RATE_LIMIT_BARRIER * 0.95
BASE_DELAY = 0.01
MAX_RETRIES = 3
REQUEST_TIMEOUT = 10

class RateLimiter:
    """Optimized rate limiter with faster weight tracking"""
    def __init__(self):
        self.weight_1m = 0
        self.weight_total = 0
        self.last_update = time.time()
        self.time_offset = 0.0

    def update(self, weight_1m: int, weight_total: int):
        current_minute = int((time.time() + self.time_offset) / 60)
        last_update_minute = int((self.last_update + self.time_offset) / 60)
        if current_minute > last_update_minute:
            self.weight_1m = 0
        self.weight_1m = max(weight_1m, self.weight_1m)
        self.weight_total = weight_total
        self.last_update = time.time()

    def should_throttle(self) -> bool:
        return self.weight_1m > SOFT_LIMIT

    def get_delay(self) -> float:
        if self.weight_1m >= HARD_LIMIT:
            return min(5.0, max(0.1, self.seconds_to_next_full_minute()))
        elif self.weight_1m >= SOFT_LIMIT:
            return BASE_DELAY * 2
        return BASE_DELAY

    def seconds_to_next_full_minute(self) -> float:
        adjusted_timestamp = time.time() + self.time_offset
        now = datetime.fromtimestamp(adjusted_timestamp)
        return 60 - (now.second + now.microsecond / 1_000_000)

class BinanceScanner:
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._time_synced = False
        self.symbol_cache = set()
        self.trade_log = []
        self.is_running = False
        self.current_tf = None
        self.higher_tf = None
        self.notification_count = 0

    def show_notification(self, title, message):
        """Show Android notification"""
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            Context = autoclass('android.content.Context')
            NotificationManager = autoclass('android.app.NotificationManager')
            Notification = autoclass('android.app.Notification')
            NotificationBuilder = autoclass('android.app.Notification$Builder')
            Intent = autoclass('android.content.Intent')
            PendingIntent = autoclass('android.app.PendingIntent')

            # Create notification channel for Android 8+
            if autoclass('android.os.Build$VERSION').SDK_INT >= 26:
                NotificationChannel = autoclass('android.app.NotificationChannel')
                channel_id = "binance_scanner"
                channel_name = "Trading Signals"
                importance = NotificationManager.IMPORTANCE_HIGH
                channel = NotificationChannel(channel_id, channel_name, importance)
                notification_service = mActivity.getSystemService(Context.NOTIFICATION_SERVICE)
                notification_service.createNotificationChannel(channel)

            # Build notification
            intent = Intent(mActivity, PythonActivity)
            pending_intent = PendingIntent.getActivity(
                mActivity, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT)
            
            builder = NotificationBuilder(mActivity, channel_id if 'channel_id' in locals() else None)
            builder.setContentTitle(title)
            builder.setContentText(message)
            builder.setSmallIcon(mActivity.getApplicationInfo().icon)
            builder.setContentIntent(pending_intent)
            builder.setAutoCancel(True)
            builder.setPriority(Notification.PRIORITY_HIGH)
            
            notification = builder.build()
            self.notification_count += 1
            notification_service.notify(self.notification_count % 100, notification)  # Cycle through IDs 0-99
        except Exception as e:
            logger.error(f"Failed to show notification: {str(e)}")

    async def fetch_candles(self, session: aiohttp.ClientSession, symbol: str, 
                          limit: int, interval: str = None) -> Optional[pd.DataFrame]:
        if interval is None:
            interval = self.current_tf
            
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        
        try:
            async with session.get(f"{BASE_URL}{API_ENDPOINT}", params=params) as response:
                response.raise_for_status()
                used_weight_1m = int(response.headers.get("x-mbx-used-weight-1m", 0))
                used_weight_total = int(response.headers.get("x-mbx-used-weight", 0))
                self.rate_limiter.update(used_weight_1m, used_weight_total)
                
                data = await response.json()
                if not data:
                    return None
                
                arr = np.array(data, dtype=np.float64)
                df = pd.DataFrame({
                    "open_time": pd.to_datetime(arr[:, 0], unit='ms', utc=True),
                    "open": arr[:, 1],
                    "high": arr[:, 2],
                    "low": arr[:, 3],
                    "close": arr[:, 4],
                    "volume": arr[:, 5],
                    "close_time": pd.to_datetime(arr[:, 6], unit='ms', utc=True),
                })
                return df
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {str(e)}")
            return None

    def check_current_tf_ema_condition(self, close: np.ndarray) -> tuple:
        """Check EMA conditions for current timeframe"""
        ema9 = ema_indicator(close, timeperiod=CURRENT_TF_EMA9)
        ema20 = ema_indicator(close, timeperiod=CURRENT_TF_EMA20)
        ema50 = ema_indicator(close, timeperiod=CURRENT_TF_EMA50)
        
        # For buy: ema9 > ema20 > ema50 and price > ema9
        buy_condition = (ema9[-1] > ema20[-1]) & (ema20[-1] > ema50[-1]) & (close[-1] > ema9[-1])
        
        # For sell: ema9 < ema20 < ema50 and price < ema9
        sell_condition = (ema9[-1] < ema20[-1]) & (ema20[-1] < ema50[-1]) & (close[-1] < ema9[-1])
        
        return buy_condition, sell_condition

    async def check_higher_tf_condition(self, session: aiohttp.ClientSession, symbol: str, signal_type: str) -> bool:
        """Check higher timeframe EMA condition"""
        df = await self.fetch_candles(session, symbol, HIGHER_TF_CANDLES, self.higher_tf)
        if df is None or len(df) < HIGHER_TF_CANDLES:
            return False
            
        close = df['close'].values
        ema = ema_indicator(close, timeperiod=HIGHER_TF_EMA)
        
        if signal_type == "BUY":
            return close[-1] > ema[-1]
        else:  # SELL
            return close[-1] < ema[-1]

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        indicators = {}
        
        # Calculate KRI (used by all combinations)
        sma = sma(close, timeperiod=KRI_MA_PERIOD)
        kri = ((close - sma) / sma) * 100
        indicators['kri'] = kri
        
        # Check current timeframe EMA conditions if enabled
        if USE_CURRENT_TF_EMA:
            current_tf_buy_cond, current_tf_sell_cond = self.check_current_tf_ema_condition(close)
            indicators['current_tf_buy_cond'] = current_tf_buy_cond
            indicators['current_tf_sell_cond'] = current_tf_sell_cond
        
        # RSI+KRI Combination
        if USE_RSI_KRI:
            rsi = rsi(close, timeperiod=14)
            indicators['rsi'] = rsi
            
            # Buy: RSI < OVERSOLD AND KRI crosses above LOWER_RSI
            kri_buy = (kri > KRI_LOWER_RSI) & (np.roll(kri, 1) <= KRI_LOWER_RSI)
            rsi_buy = rsi < RSI_OVERSOLD
            indicators['rsi_kri_buy'] = kri_buy & rsi_buy
            
            # Sell: RSI > OVERBOUGHT AND KRI crosses below UPPER_RSI
            kri_sell = (kri < KRI_UPPER_RSI) & (np.roll(kri, 1) >= KRI_UPPER_RSI)
            rsi_sell = rsi > RSI_OVERBOUGHT
            indicators['rsi_kri_sell'] = kri_sell & rsi_sell
        
        # EMA+KRI Combination
        if USE_EMA_KRI:
            ema = ema_indicator(close, timeperiod=EMA_PERIOD)
            indicators['ema'] = ema
            
            # Buy: Price > EMA AND KRI crosses above LOWER_OTHER
            ema_buy = close > ema
            kri_buy = (kri > KRI_LOWER_OTHER) & (np.roll(kri, 1) <= KRI_LOWER_OTHER)
            indicators['ema_kri_buy'] = ema_buy & kri_buy
            
            # Sell: Price < EMA AND KRI crosses below UPPER_OTHER
            ema_sell = close < ema
            kri_sell = (kri < KRI_UPPER_OTHER) & (np.roll(kri, 1) >= KRI_UPPER_OTHER)
            indicators['ema_kri_sell'] = ema_sell & kri_sell
        
        # MACD+KRI Combination
        if USE_MACD_KRI:
            macd, macd_signal, _ = macd(close, 
                                             fastperiod=MACD_FAST,
                                             slowperiod=MACD_SLOW,
                                             signalperiod=MACD_SIGNAL)
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            
            # Buy: MACD > Signal AND KRI crosses above LOWER_OTHER
            macd_buy = macd > macd_signal
            kri_buy = (kri > KRI_LOWER_OTHER) & (np.roll(kri, 1) <= KRI_LOWER_OTHER)
            indicators['macd_kri_buy'] = macd_buy & kri_buy
            
            # Sell: MACD < Signal AND KRI crosses below UPPER_OTHER
            macd_sell = macd < macd_signal
            kri_sell = (kri < KRI_UPPER_OTHER) & (np.roll(kri, 1) >= KRI_UPPER_OTHER)
            indicators['macd_kri_sell'] = macd_sell & kri_sell
        
        # BB+KRI Combination
        if USE_BB_KRI:
            upper, middle, lower = BollingerBands(close, 
                                              timeperiod=BB_PERIOD,
                                              nbdevup=BB_STDDEV,
                                              nbdevdn=BB_STDDEV)
            indicators['bb_upper'] = upper
            indicators['bb_lower'] = lower
            
            # Buy: Price < Lower BB AND KRI crosses above LOWER_OTHER
            bb_buy = close < lower
            kri_buy = (kri > KRI_LOWER_OTHER) & (np.roll(kri, 1) <= KRI_LOWER_OTHER)
            indicators['bb_kri_buy'] = bb_buy & kri_buy
            
            # Sell: Price > Upper BB AND KRI crosses below UPPER_OTHER
            bb_sell = close > upper
            kri_sell = (kri < KRI_UPPER_OTHER) & (np.roll(kri, 1) >= KRI_UPPER_OTHER)
            indicators['bb_kri_sell'] = bb_sell & kri_sell
        
        # Crossover+KRI Combination
        if USE_CROSSOVER_KRI:
            macd, macd_signal, _ = macd(close, 
                                             fastperiod=MACD_FAST,
                                             slowperiod=MACD_SLOW,
                                             signalperiod=MACD_SIGNAL)
            # Buy: MACD crosses above Signal AND KRI crosses above LOWER_OTHER
            crossover_buy = (macd > macd_signal) & (np.roll(macd, 1) <= np.roll(macd_signal, 1))
            kri_buy = (kri > KRI_LOWER_OTHER) & (np.roll(kri, 1) <= KRI_LOWER_OTHER)
            indicators['crossover_kri_buy'] = crossover_buy & kri_buy
            
            # Sell: MACD crosses below Signal AND KRI crosses below UPPER_OTHER
            crossover_sell = (macd < macd_signal) & (np.roll(macd, 1) >= np.roll(macd_signal, 1))
            kri_sell = (kri < KRI_UPPER_OTHER) & (np.roll(kri, 1) >= KRI_UPPER_OTHER)
            indicators['crossover_kri_sell'] = crossover_sell & kri_sell
        
        return indicators

    async def scan_symbol(self, session: aiohttp.ClientSession, symbol: str) -> bool:
        df = await self.fetch_candles(session, symbol, MIN_CANDLES_NEEDED)
        if df is None or len(df) < MIN_CANDLES_NEEDED:
            return False
        
        indicators = self.calculate_indicators(df)
        signal_found = False
        
        # Check last LOOKBACK_CANDLES for signals
        for i in range(-LOOKBACK_CANDLES, 0):
            if i < -len(df):
                continue
            
            timestamp = df.iloc[i]['open_time']
            price = df.iloc[i]['close']
            
            # Check all active combinations
            if USE_RSI_KRI:
                if indicators.get('rsi_kri_buy', [])[i]:
                    if not USE_CURRENT_TF_EMA or indicators.get('current_tf_buy_cond', False):
                        if USE_HIGHER_TF_CONFIRMATION:
                            higher_tf_confirmed = await self.check_higher_tf_condition(session, symbol, "BUY")
                            if higher_tf_confirmed:
                                self.log_trade(symbol, timestamp, "BUY", "RSI+KRI", price)
                                signal_found = True
                        else:
                            self.log_trade(symbol, timestamp, "BUY", "RSI+KRI", price)
                            signal_found = True
                            
                if indicators.get('rsi_kri_sell', [])[i]:
                    if not USE_CURRENT_TF_EMA or indicators.get('current_tf_sell_cond', False):
                        if USE_HIGHER_TF_CONFIRMATION:
                            higher_tf_confirmed = await self.check_higher_tf_condition(session, symbol, "SELL")
                            if higher_tf_confirmed:
                                self.log_trade(symbol, timestamp, "SELL", "RSI+KRI", price)
                                signal_found = True
                        else:
                            self.log_trade(symbol, timestamp, "SELL", "RSI+KRI", price)
                            signal_found = True
            
            if USE_EMA_KRI:
                if indicators.get('ema_kri_buy', [])[i]:
                    if not USE_CURRENT_TF_EMA or indicators.get('current_tf_buy_cond', False):
                        if USE_HIGHER_TF_CONFIRMATION:
                            higher_tf_confirmed = await self.check_higher_tf_condition(session, symbol, "BUY")
                            if higher_tf_confirmed:
                                self.log_trade(symbol, timestamp, "BUY", "EMA+KRI", price)
                                signal_found = True
                        else:
                            self.log_trade(symbol, timestamp, "BUY", "EMA+KRI", price)
                            signal_found = True
                            
                if indicators.get('ema_kri_sell', [])[i]:
                    if not USE_CURRENT_TF_EMA or indicators.get('current_tf_sell_cond', False):
                        if USE_HIGHER_TF_CONFIRMATION:
                            higher_tf_confirmed = await self.check_higher_tf_condition(session, symbol, "SELL")
                            if higher_tf_confirmed:
                                self.log_trade(symbol, timestamp, "SELL", "EMA+KRI", price)
                                signal_found = True
                        else:
                            self.log_trade(symbol, timestamp, "SELL", "EMA+KRI", price)
                            signal_found = True
            
            if USE_MACD_KRI:
                if indicators.get('macd_kri_buy', [])[i]:
                    if not USE_CURRENT_TF_EMA or indicators.get('current_tf_buy_cond', False):
                        if USE_HIGHER_TF_CONFIRMATION:
                            higher_tf_confirmed = await self.check_higher_tf_condition(session, symbol, "BUY")
                            if higher_tf_confirmed:
                                self.log_trade(symbol, timestamp, "BUY", "MACD+KRI", price)
                                signal_found = True
                        else:
                            self.log_trade(symbol, timestamp, "BUY", "MACD+KRI", price)
                            signal_found = True
                            
                if indicators.get('macd_kri_sell', [])[i]:
                    if not USE_CURRENT_TF_EMA or indicators.get('current_tf_sell_cond', False):
                        if USE_HIGHER_TF_CONFIRMATION:
                            higher_tf_confirmed = await self.check_higher_tf_condition(session, symbol, "SELL")
                            if higher_tf_confirmed:
                                self.log_trade(symbol, timestamp, "SELL", "MACD+KRI", price)
                                signal_found = True
                        else:
                            self.log_trade(symbol, timestamp, "SELL", "MACD+KRI", price)
                            signal_found = True
            
            if USE_BB_KRI:
                if indicators.get('bb_kri_buy', [])[i]:
                    if not USE_CURRENT_TF_EMA or indicators.get('current_tf_buy_cond', False):
                        if USE_HIGHER_TF_CONFIRMATION:
                            higher_tf_confirmed = await self.check_higher_tf_condition(session, symbol, "BUY")
                            if higher_tf_confirmed:
                                self.log_trade(symbol, timestamp, "BUY", "BB+KRI", price)
                                signal_found = True
                        else:
                            self.log_trade(symbol, timestamp, "BUY", "BB+KRI", price)
                            signal_found = True
                            
                if indicators.get('bb_kri_sell', [])[i]:
                    if not USE_CURRENT_TF_EMA or indicators.get('current_tf_sell_cond', False):
                        if USE_HIGHER_TF_CONFIRMATION:
                            higher_tf_confirmed = await self.check_higher_tf_condition(session, symbol, "SELL")
                            if higher_tf_confirmed:
                                self.log_trade(symbol, timestamp, "SELL", "BB+KRI", price)
                                signal_found = True
                        else:
                            self.log_trade(symbol, timestamp, "SELL", "BB+KRI", price)
                            signal_found = True
            
            if USE_CROSSOVER_KRI:
                if indicators.get('crossover_kri_buy', [])[i]:
                    if not USE_CURRENT_TF_EMA or indicators.get('current_tf_buy_cond', False):
                        if USE_HIGHER_TF_CONFIRMATION:
                            higher_tf_confirmed = await self.check_higher_tf_condition(session, symbol, "BUY")
                            if higher_tf_confirmed:
                                self.log_trade(symbol, timestamp, "BUY", "CROSSOVER+KRI", price)
                                signal_found = True
                        else:
                            self.log_trade(symbol, timestamp, "BUY", "CROSSOVER+KRI", price)
                            signal_found = True
                            
                if indicators.get('crossover_kri_sell', [])[i]:
                    if not USE_CURRENT_TF_EMA or indicators.get('current_tf_sell_cond', False):
                        if USE_HIGHER_TF_CONFIRMATION:
                            higher_tf_confirmed = await self.check_higher_tf_condition(session, symbol, "SELL")
                            if higher_tf_confirmed:
                                self.log_trade(symbol, timestamp, "SELL", "CROSSOVER+KRI", price)
                                signal_found = True
                        else:
                            self.log_trade(symbol, timestamp, "SELL", "CROSSOVER+KRI", price)
                            signal_found = True
        
        return signal_found

    def log_trade(self, symbol: str, timestamp: datetime, action: str, 
                 strategy: str, price: float):
        log_entry = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'price': price,
            'strategy': strategy
        }
        self.trade_log.append(log_entry)
        
        # Format the notification message
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        message = f"{action} {symbol} @ {price:.4f} via {strategy}"
        
        # Log to console
        logger.info(f"\n🚨 {action} SIGNAL @ {timestamp_str} | {symbol} | Price: {price:.4f} | Strategy: {strategy}")
        
        # Show Android notification
        Clock.schedule_once(lambda dt: self.show_notification(f"{action} Signal", message))

    async def run_scanner(self, current_tf: str, higher_tf: str):
        """Run the scanner with the given timeframes"""
        self.current_tf = current_tf
        self.higher_tf = higher_tf
        self.is_running = True
        
        # Hardcoded symbols list (same as in your partial code)
        symbols = [
        "BTCUSDT", "ETHUSDT", "BCHUSDT", "XRPUSDT", "EOSUSDT",
        "LTCUSDT", "TRXUSDT", "ETCUSDT", "LINKUSDT", "XLMUSDT",
        "ADAUSDT", "XMRUSDT", "DASHUSDT", "ZECUSDT", "XTZUSDT",
        "BNBUSDT", "ATOMUSDT", "ONTUSDT", "IOTAUSDT", "BATUSDT",
        "VETUSDT", "NEOUSDT", "QTUMUSDT", "IOSTUSDT", "THETAUSDT",
        "ALGOUSDT", "ZILUSDT", "KNCUSDT", "ZRXUSDT", "COMPUSDT",
        "DOGEUSDT", "SXPUSDT", "KAVAUSDT", "BANDUSDT", "RLCUSDT",
        "MKRUSDT", "SNXUSDT", "DOTUSDT", "DEFIUSDT", "YFIUSDT",
        "CRVUSDT", "TRBUSDT", "RUNEUSDT", "SUSHIUSDT", "EGLDUSDT",
        "SOLUSDT", "ICXUSDT", "STORJUSDT", "UNIUSDT", "AVAXUSDT",
        "ENJUSDT", "FLMUSDT", "KSMUSDT", "NEARUSDT", "AAVEUSDT",
        "FILUSDT", "RSRUSDT", "LRCUSDT", "BELUSDT", "AXSUSDT",
        "ALPHAUSDT", "ZENUSDT", "SKLUSDT", "GRTUSDT", "1INCHUSDT",
        "CHZUSDT", "SANDUSDT", "ANKRUSDT", "RVNUSDT", "SFPUSDT",
        "COTIUSDT", "CHRUSDT", "MANAUSDT", "ALICEUSDT", "HBARUSDT",
        "ONEUSDT", "DENTUSDT", "CELRUSDT", "HOTUSDT", "MTLUSDT",
        "OGNUSDT", "NKNUSDT", "1000SHIBUSDT", "BAKEUSDT", "GTCUSDT",
        "BTCDOMUSDT", "IOTXUSDT", "C98USDT", "MASKUSDT", "ATAUSDT",
        "DYDXUSDT", "1000XECUSDT", "GALAUSDT", "CELOUSDT", "ARUSDT",
        "ARPAUSDT", "CTSIUSDT", "LPTUSDT", "ENSUSDT", "PEOPLEUSDT",
        "ROSEUSDT", "DUSKUSDT", "FLOWUSDT", "IMXUSDT", "API3USDT",
        "GMTUSDT", "APEUSDT", "WOOUSDT", "JASMYUSDT", "OPUSDT",
        "INJUSDT", "STGUSDT", "SPELLUSDT", "1000LUNCUSDT", "LUNA2USDT",
        "LDOUSDT", "ICPUSDT", "APTUSDT", "QNTUSDT", "FETUSDT",
        "FXSUSDT", "HOOKUSDT", "MAGICUSDT", "TUSDT", "HIGHUSDT",
        "MINAUSDT", "ASTRUSDT", "PHBUSDT", "GMXUSDT", "CFXUSDT",
        "STXUSDT", "ACHUSDT", "SSVUSDT", "CKBUSDT", "PERPUSDT",
        "TRUUSDT", "LQTYUSDT", "USDCUSDT", "IDUSDT", "ARBUSDT",
        "JOEUSDT", "TLMUSDT", "LEVERUSDT", "RDNTUSDT", "HFTUSDT",
        "XVSUSDT", "BLURUSDT", "EDUUSDT", "SUIUSDT", "1000PEPEUSDT",
        "1000FLOKIUSDT", "UMAUSDT", "NMRUSDT", "MAVUSDT", "XVGUSDT",
        "WLDUSDT", "PENDLEUSDT", "ARKMUSDT", "AGLDUSDT", "YGGUSDT",
        "DODOXUSDT", "BNTUSDT", "OXTUSDT", "SEIUSDT", "CYBERUSDT",
        "HIFIUSDT", "ARKUSDT", "BICOUSDT", "BIGTIMEUSDT", "WAXPUSDT",
        "BSVUSDT", "RIFUSDT", "POLYXUSDT", "GASUSDT", "POWRUSDT",
        "TIAUSDT", "CAKEUSDT", "MEMEUSDT", "TWTUSDT", "TOKENUSDT",
        "ORDIUSDT", "STEEMUSDT", "ILVUSDT", "NTRNUSDT", "KASUSDT",
        "BEAMXUSDT", "1000BONKUSDT", "PYTHUSDT", "SUPERUSDT", "USTCUSDT",
        "ONGUSDT", "ETHWUSDT", "JTOUSDT", "1000SATSUSDT", "AUCTIONUSDT",
        "1000RATSUSDT", "ACEUSDT", "MOVRUSDT", "NFPUSDT", "AIUSDT",
        "XAIUSDT", "WIFUSDT", "MANTAUSDT", "ONDOUSDT", "LSKUSDT",
        "ALTUSDT", "JUPUSDT", "ZETAUSDT", "RONINUSDT", "DYMUSDT",
        "OMUSDT", "PIXELUSDT", "STRKUSDT", "GLMUSDT", "PORTALUSDT",
        "TONUSDT", "AXLUSDT", "MYROUSDT", "METISUSDT", "AEVOUSDT",
        "VANRYUSDT", "BOMEUSDT", "ETHFIUSDT", "ENAUSDT", "WUSDT",
        "TNSRUSDT", "SAGAUSDT", "TAOUSDT", "OMNIUSDT", "REZUSDT",
        "BBUSDT", "NOTUSDT", "TURBOUSDT", "IOUSDT", "ZKUSDT",
        "MEWUSDT", "LISTAUSDT", "ZROUSDT", "RENDERUSDT", "BANANAUSDT",
        "RAREUSDT", "GUSDT", "SYNUSDT", "SYSUSDT", "VOXELUSDT",
        "BRETTUSDT", "POPCATUSDT", "SUNUSDT", "DOGSUSDT", "MBOXUSDT",
        "CHESSUSDT", "FLUXUSDT", "BSWUSDT", "QUICKUSDT", "NEIROETHUSDT",
        "RPLUSDT", "POLUSDT", "UXLINKUSDT", "1MBABYDOGEUSDT", "NEIROUSDT",
        "KDAUSDT", "FIDAUSDT", "FIOUSDT", "CATIUSDT", "GHSTUSDT",
        "LOKAUSDT", "HMSTRUSDT", "REIUSDT", "COSUSDT", "EIGENUSDT",
        "DIAUSDT", "1000CATUSDT", "SCRUSDT", "GOATUSDT", "MOODENGUSDT",
        "SAFEUSDT", "SANTOSUSDT", "PONKEUSDT", "COWUSDT", "CETUSUSDT",
        "1000000MOGUSDT", "GRASSUSDT", "DRIFTUSDT", "SWELLUSDT", "ACTUSDT",
        "PNUTUSDT", "HIPPOUSDT", "1000XUSDT", "DEGENUSDT", "BANUSDT",
        "AKTUSDT", "SLERFUSDT", "SCRTUSDT", "1000CHEEMSUSDT", "1000WHYUSDT",
        "THEUSDT", "MORPHOUSDT", "CHILLGUYUSDT", "KAIAUSDT", "AEROUSDT",
        "ACXUSDT", "ORCAUSDT", "MOVEUSDT", "RAYSOLUSDT", "KOMAUSDT",
        "VIRTUALUSDT", "SPXUSDT", "MEUSDT", "AVAUSDT", "DEGOUSDT",
        "VELODROMEUSDT", "MOCAUSDT", "VANAUSDT", "PENGUUSDT", "LUMIAUSDT",
        "USUALUSDT", "AIXBTUSDT", "FARTCOINUSDT", "KMNOUSDT", "CGPTUSDT",
        "HIVEUSDT", "DEXEUSDT", "PHAUSDT", "DFUSDT", "GRIFFAINUSDT",
        "AI16ZUSDT", "ZEREBROUSDT", "BIOUSDT", "COOKIEUSDT", "ALCHUSDT",
        "SWARMSUSDT", "SONICUSDT", "DUSDT", "PROMUSDT", "SUSDT",
        "SOLVUSDT", "ARCUSDT", "AVAAIUSDT", "TRUMPUSDT", "MELANIAUSDT",
        "VTHOUSDT", "ANIMEUSDT", "VINEUSDT", "PIPPINUSDT", "VVVUSDT",
        "BERAUSDT", "TSTUSDT", "LAYERUSDT", "HEIUSDT", "B3USDT",
        "IPUSDT", "GPSUSDT", "SHELLUSDT", "KAITOUSDT", "REDUSDT",
        "VICUSDT", "EPICUSDT", "BMTUSDT", "MUBARAKUSDT", "FORMUSDT",
        "BIDUSDT", "TUTUSDT", "BROCCOLI714USDT", "BROCCOLIF3BUSDT", "SIRENUSDT",
        "BANANAS31USDT", "BRUSDT", "PLUMEUSDT", "NILUSDT", "PARTIUSDT",
        "JELLYJELLYUSDT", "MAVIAUSDT", "PAXGUSDT", "WALUSDT", "FUNUSDT",
        "MLNUSDT", "GUNUSDT", "ATHUSDT", "BABYUSDT", "FORTHUSDT",
        "PROMPTUSDT", "XCNUSDT", "PUMPUSDT", "STOUSDT", "FHEUSDT",
        "KERNELUSDT", "WCTUSDT", "INITUSDT", "AERGOUSDT", "BANKUSDT",
        "EPTUSDT", "DEEPUSDT", "HYPERUSDT", "MEMEFIUSDT", "FISUSDT",
        "JSTUSDT", "SIGNUSDT", "PUNDIXUSDT", "CTKUSDT", "AIOTUSDT",
        "DOLOUSDT", "HAEDALUSDT", "SXTUSDT", "ASRUSDT", "ALPINEUSDT",
        "B2USDT", "MILKUSDT", "SYRUPUSDT", "OBOLUSDT", "DOODUSDT",
        "OGUSDT", "ZKJUSDT", "SKYAIUSDT"
        ]
        
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            while self.is_running:
                start_time = time.time()
                logger.info(f"Starting scan of {len(symbols)} symbols...")
                
                # Process symbols in batches
                batch_size = 100
                for i in range(0, len(symbols), batch_size):
                    batch = symbols[i:i + batch_size]
                    tasks = [self.scan_symbol(session, sym) for sym in batch]
                    await asyncio.gather(*tasks)
                    await asyncio.sleep(0.2)  # Small delay between batches
                
                scan_time = time.time() - start_time
                wait_time = max(40 - scan_time, 1)
                logger.info(f"Scan completed in {scan_time:.2f}s. Next scan in {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

    def stop_scanner(self):
        """Stop the scanner"""
        self.is_running = False

    def cleanup_old_logs(self):
        """Clean up old trade logs to prevent memory issues"""
        if len(self.trade_log) > 50:
            self.trade_log = self.trade_log[-50:]