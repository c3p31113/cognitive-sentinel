# ==============================================================================
# Product Prototype: Cognitive Sentinel Live Monitor
# ==============================================================================
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from collections import deque
from src.sentinel import CognitiveSentinel

class LiveMonitor:
    def __init__(self, domain='phys', window_size=20):
        self.domain = domain
        self.sentinel = CognitiveSentinel(domain=domain)
        self.buffer = deque(maxlen=window_size) 
        self.window_size = window_size
        self.is_ready = False
        print(f"⚙️  [Init] Monitor initialized for domain: '{domain}'")

    def load_model(self, X_train, y_train):
        print("-" * 60)
        print("📥 [Calibration] Loading historical data for calibration...")
        self.sentinel.fit(X_train, y_train)
        self.is_ready = True
        print("✅ [Ready] System Calibrated. Invariants extracted.")
        print("-" * 60 + "\n")

    def process_stream(self, value):
        if not self.is_ready: return

        current_data = {'Sensor': value}
        self.buffer.append(current_data)
        
        buffer_status = f"[{len(self.buffer)}/{self.window_size}]"
        
        if len(self.buffer) < self.window_size:
            print(f"⏳ {buffer_status} Buffering... (Val: {value:.2f})")
            return "Buffering"

        df_window = pd.DataFrame(list(self.buffer))
        
        try:
            pred = self.sentinel.predict(df_window)[-1]
            if pred == 1:
                msg = f"🚨 [ALERT] ANOMALY DETECTED! Value: {value:.2f} (Physical Violation)"
                print(msg)
            else:
                msg = f"🟢 [Normal] System Stable.   Value: {value:.2f}"
                print(msg)
        except Exception as e:
            print(f"❌ [Error] {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   🛡️  COGNITIVE SENTINEL - LIVE MONITOR PROTOTYPE  🛡️")
    print("="*60 + "\n")

    monitor = LiveMonitor(domain='phys', window_size=20)
    
    # Calibration
    dummy_train = pd.DataFrame({'Sensor': np.random.normal(50, 2, 1000)})
    dummy_labels = np.zeros(1000)
    monitor.load_model(dummy_train, dummy_labels)
    
    print("▶️  Starting Real-time Monitoring Stream...\n")
    time.sleep(1)

    print("--- [Scenario 1] Normal Operation ---")
    normal_values = [48.5, 51.2, 49.8, 50.5, 49.1, 50.3, 49.5, 50.2]
    for v in normal_values:
        monitor.process_stream(v)
        time.sleep(0.1)

    print("\n")
    print("--- [Scenario 2] Attack Injection (Freeze Attack) ---")
    print("   ! Intruder injects fixed value to spoof sensor...")
    
    # データ数を増やして、確実に検知フェーズに入るようにする
    attack_values = [50.0] * 35 
    
    for v in attack_values:
        monitor.process_stream(v)
        time.sleep(0.05)

    print("\n" + "="*60)
    print("🏁 Demo Session Complete.")
    print("="*60)
