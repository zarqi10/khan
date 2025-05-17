from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget # For spacing
from kivy.clock import Clock
from scanner_service import ScannerService # Assuming scanner_service.py is in the same directory
from android.permissions import request_permissions, Permission # type: ignore
# import asyncio # Not directly used in main.py for async operations

class BinanceScannerApp(App):
    def build(self):
        self.service = ScannerService() # ScannerService now instantiates BinanceScanner
        
        # Main layout with ScrollView for potentially many options
        root_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        scroll_view = ScrollView(size_hint=(1, 1))
        self.layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
        self.layout.bind(minimum_height=self.layout.setter('height'))

        # Request Android permissions
        self.request_android_permissions()
        
        # --- Timeframe inputs ---
        timeframe_layout = GridLayout(cols=2, spacing=10, size_hint_y=None, height=80)
        timeframe_layout.add_widget(Label(text="Current TF:"))
        self.current_tf_input = TextInput(text='15m', multiline=False)
        timeframe_layout.add_widget(self.current_tf_input)
        
        timeframe_layout.add_widget(Label(text="Higher TF:"))
        self.higher_tf_input = TextInput(text='1h', multiline=False)
        timeframe_layout.add_widget(self.higher_tf_input)
        self.layout.add_widget(timeframe_layout)

        self.layout.add_widget(Label(text="Strategy Switches:", size_hint_y=None, height=40))
        # --- Strategy toggles ---
        strategy_layout = GridLayout(cols=2, spacing=5, size_hint_y=None)
        strategy_layout.bind(minimum_height=strategy_layout.setter('height'))

        self.strategy_toggles = {
            'USE_RSI_KRI': ToggleButton(text='RSI+KRI', group='strategies', state='down', size_hint_y=None, height=44),
            'USE_EMA_KRI': ToggleButton(text='EMA+KRI', group='strategies', state='down', size_hint_y=None, height=44),
            'USE_MACD_KRI': ToggleButton(text='MACD+KRI', group='strategies', state='down', size_hint_y=None, height=44),
            'USE_BB_KRI': ToggleButton(text='BB+KRI', group='strategies', state='down', size_hint_y=None, height=44),
            'USE_CROSSOVER_KRI': ToggleButton(text='CROSSOVER+KRI', group='strategies', state='down', size_hint_y=None, height=44)
        }
        for key, toggle_button in self.strategy_toggles.items():
            strategy_layout.add_widget(toggle_button)
        self.layout.add_widget(strategy_layout)

        self.layout.add_widget(Label(text="EMA Condition Switches:", size_hint_y=None, height=40))
        # --- EMA Condition Toggles ---
        ema_config_layout = GridLayout(cols=2, spacing=5, size_hint_y=None)
        ema_config_layout.bind(minimum_height=ema_config_layout.setter('height'))

        self.ema_condition_toggles = {
            'USE_CURRENT_TF_EMA': ToggleButton(text='Current TF EMA Cond.', state='down', size_hint_y=None, height=44),
            'USE_HIGHER_TF_CONFIRMATION': ToggleButton(text='Higher TF EMA Confirm.', state='down', size_hint_y=None, height=44)
        }
        for key, toggle_button in self.ema_condition_toggles.items():
            ema_config_layout.add_widget(toggle_button)
        self.layout.add_widget(ema_config_layout)

        self.layout.add_widget(Label(text="EMA Parameters:", size_hint_y=None, height=40))
        # --- EMA Parameters Inputs ---
        ema_params_layout = GridLayout(cols=2, spacing=10, size_hint_y=None)
        ema_params_layout.bind(minimum_height=ema_params_layout.setter('height'))

        self.ema_param_inputs = {
            'CURRENT_TF_EMA9': TextInput(text=str(self.service.scanner.CURRENT_TF_EMA9), multiline=False, input_filter='int'),
            'CURRENT_TF_EMA20': TextInput(text=str(self.service.scanner.CURRENT_TF_EMA20), multiline=False, input_filter='int'),
            'CURRENT_TF_EMA50': TextInput(text=str(self.service.scanner.CURRENT_TF_EMA50), multiline=False, input_filter='int'),
            'HIGHER_TF_EMA': TextInput(text=str(self.service.scanner.HIGHER_TF_EMA), multiline=False, input_filter='int'),
        }
        ema_params_layout.add_widget(Label(text="Curr. TF EMA9:"))
        ema_params_layout.add_widget(self.ema_param_inputs['CURRENT_TF_EMA9'])
        ema_params_layout.add_widget(Label(text="Curr. TF EMA20:"))
        ema_params_layout.add_widget(self.ema_param_inputs['CURRENT_TF_EMA20'])
        ema_params_layout.add_widget(Label(text="Curr. TF EMA50:"))
        ema_params_layout.add_widget(self.ema_param_inputs['CURRENT_TF_EMA50'])
        ema_params_layout.add_widget(Label(text="Higher TF EMA:"))
        ema_params_layout.add_widget(self.ema_param_inputs['HIGHER_TF_EMA'])
        self.layout.add_widget(ema_params_layout)

        # --- Control buttons ---
        self.control_btn = Button(text='Start Scanner', size_hint_y=None, height=60)
        self.control_btn.bind(on_press=self.toggle_scanner)
        self.layout.add_widget(self.control_btn)
        
        # --- Status label ---
        self.status_label = Label(text='Scanner ready.', size_hint_y=None, height=40)
        self.layout.add_widget(self.status_label)

        # Spacer to push content up if not enough to fill scrollview
        self.layout.add_widget(Widget(size_hint_y=None, height=50)) 

        scroll_view.add_widget(self.layout)
        root_layout.add_widget(scroll_view)
        
        # Schedule periodic cleanup
        Clock.schedule_interval(lambda dt: self.service.cleanup_old_logs(), 60) # 60 seconds
        
        return root_layout

    def request_android_permissions(self):
        permissions_to_request = [
            Permission.INTERNET,
            Permission.FOREGROUND_SERVICE,
            Permission.WAKE_LOCK,
            # Permission.RECEIVE_BOOT_COMPLETED, # Only if you implement boot receiver
        ]
        # POST_NOTIFICATIONS is for Android 13 (API 33) and above
        try:
            if int(autoclass('android.os.Build$VERSION').SDK_INT) >= 33: # type: ignore
                permissions_to_request.append(Permission.POST_NOTIFICATIONS)
        except Exception as e:
            print(f"Could not check SDK version for POST_NOTIFICATIONS: {e}")

        request_permissions(permissions_to_request, self.permissions_callback)

    def permissions_callback(self, permissions, grants):
        granted_all = all(grants)
        if not granted_all:
            self.status_label.text = "Some permissions denied. App may not function fully."
            # Optionally, show a more prominent error or guide user to settings
            print("Permissions denied:", [permissions[i] for i, granted in enumerate(grants) if not granted])
        else:
            self.status_label.text = "Permissions granted. Scanner ready."
            print("All required permissions granted.")

    def set_input_fields_disabled(self, disabled_state: bool):
        self.current_tf_input.disabled = disabled_state
        self.higher_tf_input.disabled = disabled_state
        for toggle in self.strategy_toggles.values():
            toggle.disabled = disabled_state
        for toggle in self.ema_condition_toggles.values():
            toggle.disabled = disabled_state
        for text_input in self.ema_param_inputs.values():
            text_input.disabled = disabled_state
            
    def update_scanner_config(self):
        scanner = self.service.scanner # Get the BinanceScanner instance

        # Update strategy switches
        scanner.USE_RSI_KRI = self.strategy_toggles['USE_RSI_KRI'].state == 'down'
        scanner.USE_EMA_KRI = self.strategy_toggles['USE_EMA_KRI'].state == 'down'
        scanner.USE_MACD_KRI = self.strategy_toggles['USE_MACD_KRI'].state == 'down'
        scanner.USE_BB_KRI = self.strategy_toggles['USE_BB_KRI'].state == 'down'
        scanner.USE_CROSSOVER_KRI = self.strategy_toggles['USE_CROSSOVER_KRI'].state == 'down'

        # Update EMA condition switches
        scanner.USE_CURRENT_TF_EMA = self.ema_condition_toggles['USE_CURRENT_TF_EMA'].state == 'down'
        scanner.USE_HIGHER_TF_CONFIRMATION = self.ema_condition_toggles['USE_HIGHER_TF_CONFIRMATION'].state == 'down'

        # Update EMA parameters
        try:
            scanner.CURRENT_TF_EMA9 = int(self.ema_param_inputs['CURRENT_TF_EMA9'].text)
            scanner.CURRENT_TF_EMA20 = int(self.ema_param_inputs['CURRENT_TF_EMA20'].text)
            scanner.CURRENT_TF_EMA50 = int(self.ema_param_inputs['CURRENT_TF_EMA50'].text)
            scanner.HIGHER_TF_EMA = int(self.ema_param_inputs['HIGHER_TF_EMA'].text)
            return True
        except ValueError:
            self.show_error("Invalid EMA parameter. Please enter numbers only.")
            return False

    def toggle_scanner(self, instance_button):
        if instance_button.text == 'Start Scanner':
            current_tf_str = self.current_tf_input.text.strip().lower()
            higher_tf_str = self.higher_tf_input.text.strip().lower()
            
            if not current_tf_str or not higher_tf_str:
                self.show_error("Please enter both current and higher timeframes.")
                return

            # Validate timeframes (basic validation, can be more robust)
            valid_tfs = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
            if current_tf_str not in valid_tfs or higher_tf_str not in valid_tfs:
                self.show_error(f"Invalid timeframe. Valid examples: {', '.join(valid_tfs[:6])}...")
                return

            if not self.update_scanner_config(): # Update config and check for errors
                return
                
            if self.service.start_scanning(current_tf_str, higher_tf_str):
                instance_button.text = 'Stop Scanner'
                self.status_label.text = f'Scanner running: {current_tf_str}/{higher_tf_str}'
                self.set_input_fields_disabled(True)
            else:
                self.status_label.text = 'Failed to start scanner. Already running?'
        else:
            if self.service.stop_scanning():
                instance_button.text = 'Start Scanner'
                self.status_label.text = 'Scanner stopped.'
                self.set_input_fields_disabled(False)
            else:
                self.status_label.text = 'Failed to stop scanner.' # Should not happen if it was running
    
    def show_error(self, message):
        popup = Popup(title='Error',
                     content=Label(text=message),
                     size_hint=(0.8, 0.4))
        popup.open()

    def on_stop(self):
        # Ensure scanner service is stopped when app closes, if it's running
        if self.service and self.service.is_running:
            print("App stopping, ensuring scanner service is stopped.")
            self.service.stop_scanning()
            if self.service.executor:
                self.service.executor.shutdownNow() # Forcefully shutdown executor

if __name__ == '__main__':
    BinanceScannerApp().run()