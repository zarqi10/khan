from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.clock import Clock
from scanner_service import ScannerService
from android.permissions import request_permissions, Permission
import asyncio

class BinanceScannerApp(App):
    def build(self):
        self.service = ScannerService()
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Request Android permissions
        request_permissions([
            Permission.INTERNET,
            Permission.FOREGROUND_SERVICE,
            Permission.WAKE_LOCK,
            Permission.RECEIVE_BOOT_COMPLETED,
            Permission.POST_NOTIFICATIONS
        ])
        
        # Timeframe inputs
        self.timeframe_layout = GridLayout(cols=2, spacing=10, size_hint_y=None, height=80)
        self.timeframe_layout.add_widget(Label(text="Current TF:"))
        self.current_tf_input = TextInput(text='15m', multiline=False)
        self.timeframe_layout.add_widget(self.current_tf_input)
        
        self.timeframe_layout.add_widget(Label(text="Higher TF:"))
        self.higher_tf_input = TextInput(text='1h', multiline=False)
        self.timeframe_layout.add_widget(self.higher_tf_input)
        self.layout.add_widget(self.timeframe_layout)
        
        # Strategy toggles
        self.strategy_layout = GridLayout(cols=2, spacing=5)
        self.strategy_toggles = {
            'RSI+KRI': ToggleButton(text='RSI+KRI', group='strategies', state='down'),
            'EMA+KRI': ToggleButton(text='EMA+KRI', group='strategies', state='down'),
            'MACD+KRI': ToggleButton(text='MACD+KRI', group='strategies', state='down'),
            'BB+KRI': ToggleButton(text='BB+KRI', group='strategies', state='down'),
            'CROSSOVER+KRI': ToggleButton(text='CROSSOVER+KRI', group='strategies', state='down')
        }
        for toggle in self.strategy_toggles.values():
            self.strategy_layout.add_widget(toggle)
        self.layout.add_widget(self.strategy_layout)
        
        # Control buttons
        self.control_btn = Button(text='Start Scanner', size_hint_y=None, height=60)
        self.control_btn.bind(on_press=self.toggle_scanner)
        self.layout.add_widget(self.control_btn)
        
        # Status label
        self.status_label = Label(text='Scanner ready', size_hint_y=None, height=40)
        self.layout.add_widget(self.status_label)
        
        # Schedule periodic cleanup
        Clock.schedule_interval(lambda dt: self.service.cleanup_old_logs(), 60)
        
        return self.layout
    
    def toggle_scanner(self, instance):
        if instance.text == 'Start Scanner':
            current_tf = self.current_tf_input.text
            higher_tf = self.higher_tf_input.text
            
            if not current_tf or not higher_tf:
                self.show_error("Please enter both timeframes")
                return
                
            if self.service.start_scanning(current_tf, higher_tf):
                instance.text = 'Stop Scanner'
                self.status_label.text = 'Scanner running...'
        else:
            if self.service.stop_scanning():
                instance.text = 'Start Scanner'
                self.status_label.text = 'Scanner stopped'
    
    def show_error(self, message):
        popup = Popup(title='Error',
                     content=Label(text=message),
                     size_hint=(0.8, 0.4))
        popup.open()

if __name__ == '__main__':
    BinanceScannerApp().run()