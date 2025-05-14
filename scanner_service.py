from kivy.app import App
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.properties import BooleanProperty, StringProperty
from binance_scanner import BinanceScanner
import asyncio
from android import AndroidService
from android.jni import Runnable
from java.util.concurrent import Executors
import time
import os
from datetime import datetime

class ScannerService:
    def __init__(self):
        self.executor = Executors.newSingleThreadExecutor()
        self.service = None
        self.scanner = BinanceScanner()
        self.is_running = False
        self.last_cleanup = time.time()

    def start_service(self):
        self.service = AndroidService('Binance Scanner', 'Scanning for trading signals...')
        self.service.start('service started')
        return self.service

    def start_scanning(self, current_tf, higher_tf):
        if not self.is_running:
            self.is_running = True
            self.start_service()
            
            class ScanTask(Runnable):
                def __init__(self, scanner, current_tf, higher_tf):
                    super().__init__()
                    self.scanner = scanner
                    self.current_tf = current_tf
                    self.higher_tf = higher_tf
                
                def run(self):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            self.scanner.run_scanner(self.current_tf, self.higher_tf)
                    except Exception as e:
                        print(f"Scanner error: {e}")
                    finally:
                        loop.close()
            
            self.executor.execute(ScanTask(self.scanner, current_tf, higher_tf))
            return True
        return False

    def stop_scanning(self):
        self.is_running = False
        if self.service:
            self.service.stop()
        return True

    def cleanup_old_logs(self):
        # Delete logs older than 2 cycles (assuming 1 cycle = ~1 minute)
        now = time.time()
        if now - self.last_cleanup > 120:  # Cleanup every 2 minutes
            self.last_cleanup = now
            if hasattr(self.scanner, 'trade_log'):
                # Keep only last 50 signals to prevent memory overload
                self.scanner.trade_log = self.scanner.trade_log[-50:]