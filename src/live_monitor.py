# ==============================================================================
# Product Prototype: Cognitive Sentinel Live Monitor
# ==============================================================================
# 目的: リアルタイムデータストリームに対する異常検知デモンストレーション。
#       ユーザーに分かりやすいログ出力と、内部状態の可視化を行います。
# ==============================================================================

import sys
import os
import time
import warnings

# ---------------------------------------------------------
# [System Config]
# 警告の抑制とパスの設定
# ---------------------------------------------------------
# デモの見た目を損なう内部ライブラリの警告（FutureWarning等）を抑制
warnings.filterwarnings("ignore")

# モジュール読み込み用のパス設定
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
        """
        初期化フェーズ: 正常データを学習し、システムの基準を作る
        """
        print("-" * 60)
        print("📥 [Calibration] Loading historical data for calibration...")
        print(f"   -> Training Data Size: {len(X_train)} samples")
        
        # 学習実行
        # ※Dojo生成に失敗しても元のデータで学習を継続する仕様
        self.sentinel.fit(X_train, y_train)
        
        self.is_ready = True
        print("✅ [Ready] System Calibrated. Invariants extracted.")
        print("-" * 60 + "\n")

    def process_stream(self, value):
        """
        リアルタイム処理: 1件ずつデータを受け取り判定する
        """
        if not self.is_ready:
            print("⚠️ [Error] System not armed. Run load_model() first.")
            return

        # 1. データを辞書型からDataFrameに変換するための準備
        # 入力が単一の数値の場合を想定
        current_data = {'Sensor': value}
        
        # 2. バッファに追加
        self.buffer.append(current_data)
        
        # バッファ状況の表示
        buffer_status = f"[{len(self.buffer)}/{self.window_size}]"
        
        # 3. データが溜まるまでは待機
        if len(self.buffer) < self.window_size:
            print(f"⏳ {buffer_status} Buffering data... (Value: {value:.2f})")
            return "Buffering"

        # 4. 推論実行
        # 最新のウィンドウ（バッファ全体）をDataFrameに変換
        df_window = pd.DataFrame(list(self.buffer))
        
        try:
            # sentinel.predict は 0(正常) か 1(異常) を返す
            # 最新のデータポイントに対する判定を取得
            pred = self.sentinel.predict(df_window)[-1]
            
            if pred == 1:
                msg = f"🚨 [ALERT] ANOMALY DETECTED! Value: {value:.2f} (Physical Violation)"
                print(msg)
                return "Anomaly"
            else:
                msg = f"🟢 [Normal] System Stable.   Value: {value:.2f}"
                print(msg)
                return "Normal"
                
        except Exception as e:
            print(f"❌ [Error] Inference failed: {e}")
            return "Error"

# --- メイン実行部 (デモンストレーション) ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   🛡️  COGNITIVE SENTINEL - LIVE MONITOR PROTOTYPE  🛡️")
    print("="*60 + "\n")

    # 1. インスタンス生成
    monitor = LiveMonitor(domain='phys', window_size=5)
    
    # 2. 学習フェーズ (キャリブレーション)
    # 正常なセンサーデータ(平均50, 標準偏差5)を1000件生成して学習
    dummy_train = pd.DataFrame({'Sensor': np.random.normal(50, 2, 1000)})
    dummy_labels = np.zeros(1000)
    monitor.load_model(dummy_train, dummy_labels)
    
    # 3. 監視フェーズ開始
    print("▶️  Starting Real-time Monitoring Stream...\n")
    time.sleep(1)

    # シナリオA: 正常な通信 (Normal)
    print("--- [Scenario 1] Normal Operation ---")
    normal_values = [48.5, 51.2, 49.8, 50.5, 49.1, 50.3]
    for v in normal_values:
        monitor.process_stream(v)
        time.sleep(0.2) # リアルタイム感を出すウェイト

    print("\n")
    
    # シナリオB: 攻撃発生 (Freeze Attack / 値の固定)
    # 値自体は「50.0」で正常範囲内だが、「変動がない」ため物理法則違反となる
    print("--- [Scenario 2] Attack Injection (Freeze Attack) ---")
    print("   ! Intruder injects fixed value to spoof sensor...")
    attack_values = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0] 
    
    for v in attack_values:
        monitor.process_stream(v)
        time.sleep(0.2)

    print("\n" + "="*60)
    print("🏁 Demo Session Complete.")
    print("="*60)
