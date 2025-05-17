from kivy.logger import Logger # Use Kivy's logger
# from kivy.app import App # Not needed here
# from kivy.clock import Clock # Not needed here
# from kivy.uix.label import Label # Not needed here
# ... other UI imports not needed in service file

from binance_scanner import BinanceScanner # Assuming binance_scanner.py is in the same directory
import asyncio
from android import AndroidService # type: ignore
from android.jni import Runnable # type: ignore
from java.util.concurrent import Executors, TimeUnit # type: ignore
import time
# import os # Not used
# from datetime import datetime # Not used

class ScannerService:
    def __init__(self):
        self.executor = Executors.newSingleThreadExecutor()
        self.android_service_instance = None # Renamed to avoid confusion with self.service which is Kivy App's
        self.scanner = BinanceScanner() # Scanner logic is now encapsulated here
        self.is_running = False
        self.scan_task_future = None # To keep a reference to the submitted task

    def start_android_service(self):
        """Starts the Android foreground service."""
        if not self.android_service_instance:
            self.android_service_instance = AndroidService('Binance Scanner', 'Scanning for trading signals...')
            self.android_service_instance.start('Service initiated by app.')
            Logger.info("ScannerService: Android foreground service started.")
        return self.android_service_instance

    def stop_android_service(self):
        """Stops the Android foreground service."""
        if self.android_service_instance:
            self.android_service_instance.stop()
            self.android_service_instance = None
            Logger.info("ScannerService: Android foreground service stopped.")

    def start_scanning(self, current_tf: str, higher_tf: str) -> bool:
        if self.is_running:
            Logger.warning("ScannerService: Start scanning called, but already running.")
            return False
        
        self.is_running = True
        self.start_android_service() # Ensure foreground service is active
        
        # The scanner instance's config (USE_flags, EMA_periods)
        # should have been set by main.py before calling this method.
        
        class ScanTask(Runnable):
            def __init__(self, scanner_instance, current_tf_val, higher_tf_val, service_ref):
                super().__init__()
                self.scanner = scanner_instance
                self.current_tf = current_tf_val
                self.higher_tf = higher_tf_val
                self.service_reference = service_ref # To update is_running

            def run(self):
                loop = None
                try:
                    Logger.info(f"ScannerService: ScanTask started for {self.current_tf}/{self.higher_tf}")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    # scanner.run_scanner is an async method
                    loop.run_until_complete(
                        self.scanner.run_scanner(self.current_tf, self.higher_tf)
                    )
                except InterruptedError: # Can be raised if thread is interrupted
                    Logger.warning("ScannerService: ScanTask was interrupted.")
                    # self.scanner.stop_scanner() # Ensure internal flag is set
                except Exception as e:
                    Logger.error(f"ScannerService: Error in ScanTask run loop: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    if loop and not loop.is_closed():
                        # Cancel all remaining tasks in the loop before closing
                        for task in asyncio.all_tasks(loop):
                            task.cancel()
                        # Wait for tasks to cancel
                        try:
                            # Gather and wait for cancellation, with a timeout
                            if asyncio.all_tasks(loop): # Check if there are tasks
                                loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
                        except Exception as ex_cancel:
                            Logger.error(f"ScannerService: Error during task cancellation: {ex_cancel}")
                        loop.close()
                    Logger.info("ScannerService: ScanTask finished and event loop closed.")
                    # This run method completes, so the thread will terminate.
                    # Update is_running status via the service reference
                    if self.service_reference:
                         self.service_reference.is_running = False # Mark as not running when task truly ends
                         # self.service_reference.stop_android_service() # Optionally stop foreground service if scan ends naturally
                         # For continuous scanning, this might not be desired unless explicitly stopped.

        # Submit the task
        # Keep a reference to the Future object if you need to cancel it later via Java Future API
        # However, stopping the asyncio loop from Python side (via self.scanner.is_running) is cleaner.
        current_scan_task = ScanTask(self.scanner, current_tf, higher_tf, self)
        self.scan_task_future = self.executor.submit(current_scan_task)
        Logger.info(f"ScannerService: ScanTask submitted to executor for {current_tf}/{higher_tf}.")
        return True

    def stop_scanning(self) -> bool:
        if not self.is_running and not (self.scan_task_future and not self.scan_task_future.isDone()):
            Logger.warning("ScannerService: Stop scanning called, but not actively running or task already completed.")
            # return False # Might still want to ensure service stops, so proceed

        Logger.info("ScannerService: Attempting to stop scanning.")
        self.is_running = False # Signal to the UI and potentially other checks
        
        if self.scanner:
            self.scanner.stop_scanner() # This should make the asyncio loop in run_scanner exit gracefully

        if self.scan_task_future:
            # Attempt to cancel the Java Future. This might interrupt the thread.
            # The effectiveness depends on how the Python code inside run() handles Thread.interrupt().
            # asyncio might not directly respond to Java thread interruption well.
            # The self.scanner.is_running flag is the primary mechanism.
            cancelled = self.scan_task_future.cancel(True) # True means interrupt if running
            Logger.info(f"ScannerService: Java Future cancel called, result: {cancelled}")
            self.scan_task_future = None

        # The executor itself is not shut down here, as it might be reused if scanner is restarted.
        # It will be shut down if the app closes (handled in main.py's on_stop).
        
        self.stop_android_service() # Stop the foreground service
        return True

    def cleanup_old_logs(self):
        # Cleanup happens on the main Kivy thread via Clock.schedule_interval
        if hasattr(self.scanner, 'trade_log'):
             # Use the cleanup logic defined in BinanceScanner itself if preferred,
             # or manage it here. The BinanceScanner one is:
             # if len(self.trade_log) > 100: self.trade_log = self.trade_log[-100:]
            self.scanner.cleanup_old_logs() # Call the method on the scanner instance

    def shutdown_executor(self):
        """Shuts down the executor service. Call when app is exiting."""
        if self.executor and not self.executor.isShutdown():
            Logger.info("ScannerService: Shutting down executor service.")
            try:
                self.executor.shutdown() # Disable new tasks from being submitted
                if not self.executor.awaitTermination(5, TimeUnit.SECONDS): # Wait for existing tasks to terminate
                    self.executor.shutdownNow() # Cancel currently executing tasks
                    if not self.executor.awaitTermination(5, TimeUnit.SECONDS):
                        Logger.error("ScannerService: Executor did not terminate.")
            except InterruptedError:
                self.executor.shutdownNow()
                Logger.warning("ScannerService: Executor shutdown was interrupted.")
                # Thread.currentThread().interrupt() # Preserve interrupt status if necessary

# Example of how main.py's on_stop could call this:
# In main.py:
# def on_stop(self):
#     if self.service:
#         self.service.stop_scanning() # Ensure scanner logic stops
#         self.service.shutdown_executor() # Clean up executor