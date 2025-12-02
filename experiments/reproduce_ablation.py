# ==============================================================================
# Experiment Script: Reproduction of Ablation Study (Fig 3)
# ==============================================================================
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.sentinel import CognitiveFeatureEngineer

def main():
    print("🧪 Starting Ablation Study: Causal Invariants Impact")
    
    # 1. Generate Data (The "Impossible for Baseline" Dataset)
    N = 10000
    t = np.linspace(0, 200, N)
    
    # Normal: Sine Wave (Low Frequency)
    # 値の範囲: -1.0 〜 1.0
    X_normal = pd.DataFrame({'Sensor_A': np.sin(t)})
    y_normal = np.zeros(N)
    
    # Attack: Sine Wave (High Frequency)
    # 値の範囲: -1.0 〜 1.0 (正常と完全に同じ！)
    # Baseline（値しか見ない）には、この2つは「同じデータの集まり」に見えるため、
    # 原理的に区別がつかず、スコアは0.5（当てずっぽう）になるはず。
    X_attack = pd.DataFrame({'Sensor_A': np.sin(t * 5.0)}) 
    y_attack = np.ones(N)
    
    # Combine
    X_all = pd.concat([X_normal, X_attack]).reset_index(drop=True)
    y_all = np.concatenate([y_normal, y_attack])
    
    # 2. Engineer Features
    print("   -> Calculating Causal Features (Before Shuffling)...")
    engineer = CognitiveFeatureEngineer()
    X_eng = engineer.process(X_all, 'phys')
    
    # 3. Split (Shuffle=True is OK here because classes are purely separated by dynamics)
    # 単純なランダム分割でも、Baselineは「値」しか見ないので勝てない。
    X_train_eng, X_test_eng, y_train, y_test = train_test_split(
        X_eng, y_all, test_size=0.3, shuffle=True, random_state=42
    )
    
    # 4. Prepare Baseline
    raw_cols = ['Sensor_A']
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

    # Save Graph
    results = pd.DataFrame({
        'Model': ['Baseline (Raw)', 'Proposed (Causal)'],
        'F1-Score': [f1_base, f1_prop]
    })
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Model', y='F1-Score', data=results, palette='viridis')
    plt.title("Ablation Study: Impact of Causal Invariants", fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for index, row in results.iterrows():
        plt.text(index, row['F1-Score'] + 0.02, f"{row['F1-Score']:.3f}", color='black', ha="center")
    
    save_path = os.path.join(os.path.dirname(__file__), 'result_ablation.png')
    plt.savefig(save_path)
    print(f"📊 Graph saved to: {save_path}")

if __name__ == "__main__":
    main()
