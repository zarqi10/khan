import asyncio
import aiohttp
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

# Technical Analysis library imports
import ta
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from kivy.clock import Clock
from android import mActivity # Assuming this is available in your Kivy-Android environment
from jnius import autoclass # Assuming this is available

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
USE_RSI_KRI = True       # RSI + KRI combination
USE_EMA_KRI = True      # EMA + KRI combination
USE_MACD_KRI = True     # MACD + KRI combination
USE_BB_KRI = True       # Bollinger Bands + KRI combination
USE_CROSSOVER_KRI = True # MACD Crossover + KRI combination

# New EMA Switch Configuration (will be set by UI)
USE_CURRENT_TF_EMA = True  # Switch for current timeframe EMA conditions
USE_HIGHER_TF_CONFIRMATION = True  # Switch for higher timeframe confirmation
CURRENT_TF_EMA9 = 5
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

        # --- Instance-level configuration (can be updated by UI) ---
        # These mirror the global USE_ flags and parameters.
        # If main.py updates these instance variables, they will be used.
        # Otherwise, the global fallbacks (defined above) will be used by functions
        # if they are not modified to use self.
        # For now, to minimize changes, functions will use globals unless
        # explicitly passed 'self' and use self.parameter.
        # This is a point for future refactoring for better encapsulation.
        self.USE_RSI_KRI = USE_RSI_KRI
        self.USE_EMA_KRI = USE_EMA_KRI
        self.USE_MACD_KRI = USE_MACD_KRI
        self.USE_BB_KRI = USE_BB_KRI
        self.USE_CROSSOVER_KRI = USE_CROSSOVER_KRI
        self.USE_CURRENT_TF_EMA = USE_CURRENT_TF_EMA
        self.USE_HIGHER_TF_CONFIRMATION = USE_HIGHER_TF_CONFIRMATION
        self.CURRENT_TF_EMA9 = CURRENT_TF_EMA9
        self.CURRENT_TF_EMA20 = CURRENT_TF_EMA20
        self.CURRENT_TF_EMA50 = CURRENT_TF_EMA50
        self.HIGHER_TF_EMA = HIGHER_TF_EMA
        # ... other parameters can also be instance variables if needed


    def show_notification(self, title, message):
        """Show Android notification"""
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            currentActivity = PythonActivity.mActivity
            Context = autoclass('android.content.Context')
            NotificationManager = autoclass('android.app.NotificationManager')
            
            # For Android 8.0 (API level 26) and higher, you must create a notification channel.
            if autoclass('android.os.Build$VERSION').SDK_INT >= 26:
                NotificationChannel = autoclass('android.app.NotificationChannel')
                channel_id = "binance_scanner_channel" # Unique channel ID
                channel_name = "Trading Signals"
                importance = NotificationManager.IMPORTANCE_HIGH # Or other importance levels
                channel = NotificationChannel(channel_id, channel_name, importance)
                channel.setDescription("Notifications for Binance trading signals")
                # Optionally, configure other channel properties (lights, vibration, etc.)
                # channel.enableLights(True)
                # channel.setLightColor(autoclass('android.graphics.Color').RED)
                # channel.enableVibration(True)
                
                notification_service = currentActivity.getSystemService(Context.NOTIFICATION_SERVICE)
                notification_service.createNotificationChannel(channel)
                NotificationBuilder = autoclass('android.app.Notification$Builder') # Pre-Oreo
                NotificationCompatBuilder = autoclass('androidx.core.app.NotificationCompat$Builder') # For modern compatibility
                builder = NotificationCompatBuilder(currentActivity, channel_id)

            else: # For older Android versions
                NotificationBuilder = autoclass('android.app.Notification$Builder')
                builder = NotificationBuilder(currentActivity)

            Intent = autoclass('android.content.Intent')
            PendingIntent = autoclass('android.app.PendingIntent')
            
            intent = Intent(currentActivity, PythonActivity) # Intent to launch app when notification is clicked
            # FLAG_IMMUTABLE or FLAG_MUTABLE is required for PendingIntent for Android 12+
            pending_intent_flag = PendingIntent.FLAG_UPDATE_CURRENT
            if autoclass('android.os.Build$VERSION').SDK_INT >= 23: # FLAG_IMMUTABLE introduced in API 23
                 pending_intent_flag |= PendingIntent.FLAG_IMMUTABLE


            pending_intent = PendingIntent.getActivity(currentActivity, 0, intent, pending_intent_flag)
            
            # Use the app's icon. Make sure it's suitable for notifications (e.g., small, alpha channel)
            app_info = currentActivity.getApplicationInfo()
            icon = app_info.icon
            if icon == 0: # Fallback if no app icon found (should not happen for a packaged app)
                icon = autoclass('android.R$drawable').sym_def_app_icon


            builder.setContentTitle(title)
            builder.setContentText(message)
            builder.setSmallIcon(icon) 
            builder.setContentIntent(pending_intent)
            builder.setAutoCancel(True) # Notification dismissed when clicked
            
            if autoclass('android.os.Build$VERSION').SDK_INT < 26: # Set priority for pre-Oreo
                 builder.setPriority(autoclass('android.app.Notification').PRIORITY_HIGH)

            notification = builder.build()
            
            notification_service = currentActivity.getSystemService(Context.NOTIFICATION_SERVICE)
            self.notification_count += 1
            notification_service.notify(self.notification_count % 100, notification)  # Cycle through IDs 0-99
            logger.info(f"Notification '{title}' sent: {message}")

        except Exception as e:
            logger.error(f"Failed to show notification: {str(e)}")
            import traceback
            traceback.print_exc()


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
            async with session.get(f"{BASE_URL}{API_ENDPOINT}", params=params, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                used_weight_1m = int(response.headers.get("x-mbx-used-weight-1m", 0))
                used_weight_total = int(response.headers.get("x-mbx-used-weight", 0))
                self.rate_limiter.update(used_weight_1m, used_weight_total)

                data = await response.json()
                if not data:
                    return None

                arr = np.array(data, dtype=np.float64)
                df = pd.DataFrame({
                    "open_time": pd.to_datetime(arr[:, 0], unit='ms', tz='UTC'), # Explicitly UTC
                    "open": arr[:, 1],
                    "high": arr[:, 2],
                    "low": arr[:, 3],
                    "close": arr[:, 4],
                    "volume": arr[:, 5],
                    "close_time": pd.to_datetime(arr[:, 6], unit='ms', tz='UTC'), # Explicitly UTC
                })
                return df
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol} on interval {interval}: {str(e)}")
            return None

    def check_current_tf_ema_condition(self, close_prices: np.ndarray) -> tuple:
        """Check EMA conditions for current timeframe using instance parameters"""
        close_series = pd.Series(close_prices)
        
        ema9_indicator = EMAIndicator(close=close_series, window=self.CURRENT_TF_EMA9)
        ema9 = ema9_indicator.ema_indicator()
        
        ema20_indicator = EMAIndicator(close=close_series, window=self.CURRENT_TF_EMA20)
        ema20 = ema20_indicator.ema_indicator()
        
        ema50_indicator = EMAIndicator(close=close_series, window=self.CURRENT_TF_EMA50)
        ema50 = ema50_indicator.ema_indicator()

        if ema9.empty or ema20.empty or ema50.empty or len(close_series) == 0:
            return False, False
        
        # For buy: ema9 > ema20 > ema50 and price > ema9
        buy_condition = (ema9.iloc[-1] > ema20.iloc[-1]) and \
                        (ema20.iloc[-1] > ema50.iloc[-1]) and \
                        (close_series.iloc[-1] > ema9.iloc[-1])
        
        # For sell: ema9 < ema20 < ema50 and price < ema9
        sell_condition = (ema9.iloc[-1] < ema20.iloc[-1]) and \
                         (ema20.iloc[-1] < ema50.iloc[-1]) and \
                         (close_series.iloc[-1] < ema9.iloc[-1])
        
        return buy_condition, sell_condition

    async def check_higher_tf_condition(self, session: aiohttp.ClientSession, symbol: str, signal_type: str) -> bool:
        """Check higher timeframe EMA condition using instance parameters"""
        df = await self.fetch_candles(session, symbol, HIGHER_TF_CANDLES, self.higher_tf) # Global HIGHER_TF_CANDLES
        if df is None or len(df) < HIGHER_TF_CANDLES: # Global HIGHER_TF_CANDLES
            return False

        close_series = df['close'] # This is already a pandas Series from fetch_candles
        
        ema_indicator_obj = EMAIndicator(close=close_series, window=self.HIGHER_TF_EMA)
        ema = ema_indicator_obj.ema_indicator()

        if ema.empty or len(close_series) == 0:
            return False
            
        if signal_type == "BUY":
            return close_series.iloc[-1] > ema.iloc[-1]
        else:  # SELL
            return close_series.iloc[-1] < ema.iloc[-1]

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Ensure 'close' is a pandas Series for 'ta' library
        close_series = df['close']
        # high_series = df['high'] # Not used by current ta indicators but good practice
        # low_series = df['low']   # Not used by current ta indicators

        indicators = {}

        # Calculate KRI (used by all combinations)
        sma_indicator = SMAIndicator(close=close_series, window=KRI_MA_PERIOD) # Global KRI_MA_PERIOD
        sma_values = sma_indicator.sma_indicator()
        kri_series = ((close_series - sma_values) / sma_values) * 100
        indicators['kri'] = kri_series.to_numpy() # Convert to NumPy for np.roll

        # Check current timeframe EMA conditions if enabled
        if self.USE_CURRENT_TF_EMA: # Using instance variable
            # Pass numpy array as original function expects
            current_tf_buy_cond, current_tf_sell_cond = self.check_current_tf_ema_condition(close_series.to_numpy())
            indicators['current_tf_buy_cond'] = current_tf_buy_cond
            indicators['current_tf_sell_cond'] = current_tf_sell_cond

        # RSI+KRI Combination
        if self.USE_RSI_KRI: # Using instance variable
            rsi_indicator = RSIIndicator(close=close_series, window=14) # Default RSI period
            rsi_values = rsi_indicator.rsi()
            indicators['rsi'] = rsi_values.to_numpy() # Convert to NumPy

            kri_buy_cond = (indicators['kri'] > KRI_LOWER_RSI) & (np.roll(indicators['kri'], 1) <= KRI_LOWER_RSI)
            rsi_buy_cond = indicators['rsi'] < RSI_OVERSOLD
            indicators['rsi_kri_buy'] = kri_buy_cond & rsi_buy_cond

            kri_sell_cond = (indicators['kri'] < KRI_UPPER_RSI) & (np.roll(indicators['kri'], 1) >= KRI_UPPER_RSI)
            rsi_sell_cond = indicators['rsi'] > RSI_OVERBOUGHT
            indicators['rsi_kri_sell'] = kri_sell_cond & rsi_sell_cond

        # EMA+KRI Combination
        if self.USE_EMA_KRI: # Using instance variable
            ema_obj = EMAIndicator(close=close_series, window=EMA_PERIOD) # Global EMA_PERIOD
            ema_values = ema_obj.ema_indicator()
            indicators['ema'] = ema_values.to_numpy() # Convert to NumPy

            # Need close as numpy array for direct comparison with ema numpy array
            close_np = close_series.to_numpy()
            ema_buy_cond = close_np > indicators['ema']
            kri_buy_cond = (indicators['kri'] > KRI_LOWER_OTHER) & (np.roll(indicators['kri'], 1) <= KRI_LOWER_OTHER)
            indicators['ema_kri_buy'] = ema_buy_cond & kri_buy_cond

            ema_sell_cond = close_np < indicators['ema']
            kri_sell_cond = (indicators['kri'] < KRI_UPPER_OTHER) & (np.roll(indicators['kri'], 1) >= KRI_UPPER_OTHER)
            indicators['ema_kri_sell'] = ema_sell_cond & kri_sell_cond

        # MACD+KRI Combination
        if self.USE_MACD_KRI: # Using instance variable
            macd_obj = MACD(close=close_series, window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL) # Globals
            indicators['macd'] = macd_obj.macd().to_numpy()
            indicators['macd_signal'] = macd_obj.macd_signal().to_numpy()

            macd_buy_cond = indicators['macd'] > indicators['macd_signal']
            kri_buy_cond = (indicators['kri'] > KRI_LOWER_OTHER) & (np.roll(indicators['kri'], 1) <= KRI_LOWER_OTHER)
            indicators['macd_kri_buy'] = macd_buy_cond & kri_buy_cond

            macd_sell_cond = indicators['macd'] < indicators['macd_signal']
            kri_sell_cond = (indicators['kri'] < KRI_UPPER_OTHER) & (np.roll(indicators['kri'], 1) >= KRI_UPPER_OTHER)
            indicators['macd_kri_sell'] = macd_sell_cond & kri_sell_cond

        # BB+KRI Combination
        if self.USE_BB_KRI: # Using instance variable
            bb_obj = BollingerBands(close=close_series, window=BB_PERIOD, window_dev=BB_STDDEV) # Globals
            indicators['bb_upper'] = bb_obj.bollinger_hband().to_numpy()
            indicators['bb_lower'] = bb_obj.bollinger_lband().to_numpy()
            
            # Need close as numpy array
            close_np = close_series.to_numpy()
            bb_buy_cond = close_np < indicators['bb_lower']
            kri_buy_cond = (indicators['kri'] > KRI_LOWER_OTHER) & (np.roll(indicators['kri'], 1) <= KRI_LOWER_OTHER)
            indicators['bb_kri_buy'] = bb_buy_cond & kri_buy_cond

            bb_sell_cond = close_np > indicators['bb_upper']
            kri_sell_cond = (indicators['kri'] < KRI_UPPER_OTHER) & (np.roll(indicators['kri'], 1) >= KRI_UPPER_OTHER)
            indicators['bb_kri_sell'] = bb_sell_cond & kri_sell_cond

        # Crossover+KRI Combination
        if self.USE_CROSSOVER_KRI: # Using instance variable
            # MACD already calculated if USE_MACD_KRI is true, otherwise calculate again
            if 'macd' not in indicators or 'macd_signal' not in indicators:
                 macd_obj = MACD(close=close_series, window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL) # Globals
                 indicators['macd'] = macd_obj.macd().to_numpy()
                 indicators['macd_signal'] = macd_obj.macd_signal().to_numpy()

            crossover_buy_cond = (indicators['macd'] > indicators['macd_signal']) & \
                               (np.roll(indicators['macd'], 1) <= np.roll(indicators['macd_signal'], 1))
            kri_buy_cond = (indicators['kri'] > KRI_LOWER_OTHER) & (np.roll(indicators['kri'], 1) <= KRI_LOWER_OTHER)
            indicators['crossover_kri_buy'] = crossover_buy_cond & kri_buy_cond

            crossover_sell_cond = (indicators['macd'] < indicators['macd_signal']) & \
                                (np.roll(indicators['macd'], 1) >= np.roll(indicators['macd_signal'], 1))
            kri_sell_cond = (indicators['kri'] < KRI_UPPER_OTHER) & (np.roll(indicators['kri'], 1) >= KRI_UPPER_OTHER)
            indicators['crossover_kri_sell'] = crossover_sell_cond & kri_sell_cond
            
        return indicators

    async def scan_symbol(self, session: aiohttp.ClientSession, symbol: str) -> bool:
        df = await self.fetch_candles(session, symbol, MIN_CANDLES_NEEDED) # Global MIN_CANDLES_NEEDED
        if df is None or len(df) < MIN_CANDLES_NEEDED: # Global MIN_CANDLES_NEEDED
            return False

        # Fill NaNs that might come from indicator calculations at the start of the series
        # This is important because np.roll on NaN arrays or indexing might lead to unexpected behavior or errors
        # A simple forward fill is often used, or you can ensure enough data for all indicators to be non-NaN
        df = df.ffill().bfill() # Basic fill, might need more sophisticated handling

        indicators = self.calculate_indicators(df)
        signal_found = False

        # Check last LOOKBACK_CANDLES for signals
        for i in range(-LOOKBACK_CANDLES, 0): # Global LOOKBACK_CANDLES
            # Ensure index is valid for the indicator arrays (which have same length as df after calculation)
            if i < -len(df): # Should be equivalent to i + len(df) < 0
                continue
            
            # Check if any key indicator data is NaN for this candle `i`
            # This is crucial because calculations on NaNs (e.g. NaN > 5) result in False, not an error,
            # but it's better to be explicit or ensure data is clean.
            # For example, if indicators['kri'][i] is NaN, conditions involving it will be False.

            timestamp = df.iloc[i]['open_time']
            price = df.iloc[i]['close']

            # --- BUY SIGNALS ---
            buy_strategies_triggered = []
            if self.USE_RSI_KRI and indicators.get('rsi_kri_buy', [])[i]: buy_strategies_triggered.append("RSI+KRI")
            if self.USE_EMA_KRI and indicators.get('ema_kri_buy', [])[i]: buy_strategies_triggered.append("EMA+KRI")
            if self.USE_MACD_KRI and indicators.get('macd_kri_buy', [])[i]: buy_strategies_triggered.append("MACD+KRI")
            if self.USE_BB_KRI and indicators.get('bb_kri_buy', [])[i]: buy_strategies_triggered.append("BB+KRI")
            if self.USE_CROSSOVER_KRI and indicators.get('crossover_kri_buy', [])[i]: buy_strategies_triggered.append("CROSSOVER+KRI")

            for strategy_name in buy_strategies_triggered:
                passes_current_tf_ema = not self.USE_CURRENT_TF_EMA or indicators.get('current_tf_buy_cond', False)
                if passes_current_tf_ema:
                    passes_higher_tf = True # Assume true if not used
                    if self.USE_HIGHER_TF_CONFIRMATION:
                        passes_higher_tf = await self.check_higher_tf_condition(session, symbol, "BUY")
                    
                    if passes_higher_tf:
                        self.log_trade(symbol, timestamp, "BUY", strategy_name, price)
                        signal_found = True
                        break # Found a buy signal for this candle, move to next candle or symbol

            if signal_found and buy_strategies_triggered: continue # Move to next candle if buy found

            # --- SELL SIGNALS ---
            sell_strategies_triggered = []
            if self.USE_RSI_KRI and indicators.get('rsi_kri_sell', [])[i]: sell_strategies_triggered.append("RSI+KRI")
            if self.USE_EMA_KRI and indicators.get('ema_kri_sell', [])[i]: sell_strategies_triggered.append("EMA+KRI")
            if self.USE_MACD_KRI and indicators.get('macd_kri_sell', [])[i]: sell_strategies_triggered.append("MACD+KRI")
            if self.USE_BB_KRI and indicators.get('bb_kri_sell', [])[i]: sell_strategies_triggered.append("BB+KRI")
            if self.USE_CROSSOVER_KRI and indicators.get('crossover_kri_sell', [])[i]: sell_strategies_triggered.append("CROSSOVER+KRI")

            for strategy_name in sell_strategies_triggered:
                passes_current_tf_ema = not self.USE_CURRENT_TF_EMA or indicators.get('current_tf_sell_cond', False)
                if passes_current_tf_ema:
                    passes_higher_tf = True # Assume true if not used
                    if self.USE_HIGHER_TF_CONFIRMATION:
                        passes_higher_tf = await self.check_higher_tf_condition(session, symbol, "SELL")
                        
                    if passes_higher_tf:
                        self.log_trade(symbol, timestamp, "SELL", strategy_name, price)
                        signal_found = True
                        break # Found a sell signal for this candle

            if signal_found and sell_strategies_triggered : continue # Move to next candle if sell found
            
        return signal_found

    def log_trade(self, symbol: str, timestamp: pd.Timestamp, action: str,
                 strategy: str, price: float):
        # Ensure timestamp is timezone-aware (it should be from fetch_candles)
        # Convert to string for logging and notification
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        log_entry = {
            'timestamp': timestamp_str, # Store as string
            'symbol': symbol,
            'action': action,
            'price': price,
            'strategy': strategy
        }
        self.trade_log.append(log_entry)

        message = f"{action} {symbol} @ {price:.4f} via {strategy}"
        logger.info(f"\n🚨 {action} SIGNAL @ {timestamp_str} | {symbol} | Price: {price:.4f} | Strategy: {strategy}")
        
        # Schedule notification on Kivy's main thread
        Clock.schedule_once(lambda dt: self.show_notification(f"{action} Signal ({strategy})", message))


    async def run_scanner(self, current_tf: str, higher_tf: str):
        """Run the scanner with the given timeframes"""
        self.current_tf = current_tf
        self.higher_tf = higher_tf
        self.is_running = True

        # SYMBOLS LIST - Add your hardcoded list here or load from a file
        # For example:
        # symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", ...]
        # Or load from a file included in your APK:
        # try:
        #     with open("symbols.txt", "r") as f:
        #         symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
        # except FileNotFoundError:
        #     logger.error("symbols.txt not found. Please create it.")
        #     symbols = ["BTCUSDT", "ETHUSDT"] # Fallback
        # except Exception as e:
        #     logger.error(f"Error loading symbols.txt: {e}")
        #     symbols = ["BTCUSDT", "ETHUSDT"] # Fallback
        
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
        if not symbols or not symbols[0].endswith("USDT"): # Basic check
             logger.warning("Symbols list might be empty or not properly set. Using default BTC/ETH.")
             symbols = ["BTCUSDT", "ETHUSDT"]


        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, ssl=False) # ssl=False can help with some SSL issues on Android sometimes, but use with caution.
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            while self.is_running:
                start_time = time.time()
                logger.info(f"Starting scan of {len(symbols)} symbols for TF {self.current_tf} (Higher TF: {self.higher_tf})...")

                batch_size = 50 # Reduced batch size for potentially slower mobile network
                for i in range(0, len(symbols), batch_size):
                    if not self.is_running: break # Check if scanner was stopped
                    
                    batch = symbols[i:i + batch_size]
                    tasks = []
                    for sym in batch:
                        if not self.is_running: break
                        # Apply rate limiting delay before each request if needed by rate_limiter logic
                        if self.rate_limiter.should_throttle():
                            delay = self.rate_limiter.get_delay()
                            logger.warning(f"Rate limit approaching. Delaying for {delay:.2f}s. Weight: {self.rate_limiter.weight_1m}")
                            await asyncio.sleep(delay)
                        tasks.append(self.scan_symbol(session, sym))
                    
                    if tasks:
                        await asyncio.gather(*tasks)
                    
                    if i + batch_size < len(symbols) and self.is_running : # Avoid sleep after last batch or if stopped
                         await asyncio.sleep(0.2) # Small delay between batches

                if not self.is_running:
                    logger.info("Scanner loop interrupted.")
                    break

                scan_time = time.time() - start_time
                # Determine wait time based on current timeframe. Minimum 1 min for '1m' TF.
                # Example: For 1m TF, scan every ~60s. For 5m TF, scan every ~300s.
                # This is a simple approach; more sophisticated scheduling might be needed.
                interval_seconds = 60 # Default to 1 minute
                if self.current_tf.endswith('m'):
                    try: interval_seconds = int(self.current_tf[:-1]) * 60
                    except ValueError: pass
                elif self.current_tf.endswith('h'):
                    try: interval_seconds = int(self.current_tf[:-1]) * 3600
                    except ValueError: pass
                elif self.current_tf.endswith('d'):
                    try: interval_seconds = int(self.current_tf[:-1]) * 86400
                    except ValueError: pass
                
                wait_time = max(1.0, interval_seconds - scan_time) # Ensure at least 1s wait
                logger.info(f"Scan completed in {scan_time:.2f}s. Next scan for {self.current_tf} in {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
            
            logger.info("Scanner run loop finished.")


    def stop_scanner(self):
        """Stop the scanner"""
        logger.info("Stop scanner requested.")
        self.is_running = False

    def cleanup_old_logs(self):
        """Clean up old trade logs to prevent memory issues"""
        if len(self.trade_log) > 100: # Increased slightly
            self.trade_log = self.trade_log[-100:] # Keep last 100
            logger.info("Cleaned up old trade logs.")