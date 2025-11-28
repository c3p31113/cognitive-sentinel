# ==============================================================================
# Experiment: Ablation Study for Causal Invariants
# ==============================================================================
# This script reproduces Figure 3 in the paper.
# It compares "Raw Features + LightGBM" vs "Causal Features + LightGBM".
# ==============================================================================

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

# Path adjustment to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.sentinel import CognitiveFeatureEngineer, AdversarialSimulator

def main():
    print("🧪 Starting Ablation Study: Causal Invariants Impact")
    
    # 1. Generate Data (Physics Simulation)
    N = 6000
    t = np.linspace(0, 80, N)
    
    # Normal: Slow Sine Wave
    X_normal = pd.DataFrame({'Sensor_A': np.sin(t) + np.random.normal(0, 0.05, N)})
    y_normal = np.zeros(N)
    
    # Attack: Frequency Attack (High Vibe) using Simulator
    sim = AdversarialSimulator()
    X_attack = sim.inject_attack(X_normal.copy(), 'phys') # Inject frequency attack
    # Note: Simulator injects into the whole DF, we assume half is attacked for this demo setup
    # Manually ensuring the attack structure for clean experiment:
    attack_start = N // 2
    X_attack = X_attack.iloc[attack_start:].reset_index(drop=True)
    y_attack = np.ones(len(X_attack))
    
    X_normal = X_normal.iloc[:attack_start].reset_index(drop=True)
    y_normal = y_normal[:attack_start]
    
    # Combine
    X_all = pd.concat([X_normal, X_attack]).reset_index(drop=True)
    y_all = np.concatenate([y_normal, y_attack])
    
    # 2. Engineer Features (BEFORE SPLIT)
    engineer = CognitiveFeatureEngineer()
    X_eng = engineer.process(X_all, 'phys')
    
    # 3. Split
    X_train_eng, X_test_eng, y_train, y_test = train_test_split(
        X_eng, y_all, test_size=0.3, shuffle=True, random_state=42
    )
    
    # 4. Prepare Baseline (Raw Only)
    # Remove engineered columns (lag, roll, dev)
    raw_cols = [c for c in X_train_eng.columns if '_' not in c]
    X_train_raw = X_train_eng[raw_cols]
    X_test_raw = X_test_eng[raw_cols]
    
    # 5. Train & Evaluate
    print("\nTraining Baseline (Raw)...")
    clf_base = LGBMClassifier(random_state=42, verbose=-1)
    clf_base.fit(X_train_raw, y_train)
    pred_base = clf_base.predict(X_test_raw)
    f1_base = f1_score(y_test, pred_base)
    
    print("\nTraining Proposed (Causal)...")
    clf_prop = LGBMClassifier(random_state=42, verbose=-1)
    clf_prop.fit(X_train_eng, y_train)
    pred_prop = clf_prop.predict(X_test_eng)
    f1_prop = f1_score(y_test, pred_prop)
    
    # 6. Report
    print("-" * 30)
    print(f"RESULTS (F1-Score):")
    print(f"Baseline: {f1_base:.4f}")
    print(f"Proposed: {f1_prop:.4f}")
    print(f"Diff    : {f1_prop - f1_base:+.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()