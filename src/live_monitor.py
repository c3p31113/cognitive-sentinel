# ==============================================================================
# Product Prototype: Cognitive Sentinel Live Monitor
# ==============================================================================
# 目的: CSVに依存せず、リアルタイムに流れてくるデータ（ストリーム）を
#       1件ずつ処理し、異常を即時判定する「製品版」のモックアップ。
# ==============================================================================

import numpy as np
import pandas as pd
from collections import deque
from src.sentinel import CognitiveSentinel

class LiveMonitor:
    def __init__(self, domain='phys', window_size=20):
        self.domain = domain
        self.sentinel = CognitiveSentinel(domain=domain)
        self.buffer = deque(maxlen=window_size) # ストリーム用バッファ
        self.is_ready = False

    def load_model(self, X_train, y_train):
        """
        初期化：正常な環境データを読み込んで「基準」を作る
        （実運用では、最初の1時間は学習モード、その後監視モードにする等）
        """
        print(f"🔵 [System] Calibrating for {self.domain} environment...")
        self.sentinel.fit(X_train, y_train)
        self.is_ready = True
        print("🟢 [System] System Armed. Ready to detect.")

    def process_stream(self, incoming_data_point):
        """
        リアルタイム処理：データが1件来るたびに判定する
        input: {'Sensor': 0.5, ...} (辞書型)
        """
        if not self.is_ready: return "Initializing..."

        # 1. バッファに追加（時系列の特徴量を作るため）
        self.buffer.append(incoming_data_point)
        if len(self.buffer) < 5: return "Buffering..." # データが溜まるまで待機

        # 2. DataFrameに変換（1行だけのDF）
        df_current = pd.DataFrame(list(self.buffer))
        
        # 3. 最新の1行だけを判定
        # (sentinel内部で特徴量計算 -> 判定まで一気に行う)
        # ※最新行の判定には過去のバッファが必要なのでdf_currentを渡す
        pred = self.sentinel.predict(df_current)[-1] 
        
        if pred == 1:
            return "🚨 ALERT: Anomaly Detected!"
        else:
            return "✅ Normal"

# --- デモ実行 (Usage Example) ---
if __name__ == "__main__":
    # 1. 仮想のセンサー (サーバーCPU温度計だとする)
    monitor = LiveMonitor('phys')
    
    # 2. 学習フェーズ (正常な環境音を聞かせる)
    # 本来は過去ログなどを食わせる
    print("\n--- Phase 1: Learning Normal Behavior ---")
    dummy_train = pd.DataFrame({'Sensor': np.random.normal(50, 5, 1000)})
    monitor.load_model(dummy_train, np.zeros(1000))
    
    # 3. 運用フェーズ (データが1秒に1回来ると想定)
    print("\n--- Phase 2: Real-time Monitoring ---")
    
    # 正常なデータが流れてくる...
    for i in range(3):
        val = np.random.normal(50, 5)
        status = monitor.process_stream({'Sensor': val})
        print(f"Input: {val:.2f} -> {status}")
        
    # 突然、攻撃発生！ (Freeze攻撃: 値が固まる)
    print("\n!! ATTACK STARTED !!")
    fixed_val = 52.0
    for i in range(3):
        status = monitor.process_stream({'Sensor': fixed_val})
        print(f"Input: {fixed_val:.2f} -> {status}")