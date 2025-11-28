# ==============================================================================
# Experiment Script: Reproduction of Ablation Study (Fig 3)
# ==============================================================================
# 目的: 本論文 Section 5.7 で議論されたアブレーション研究を再現する。
#       「生データのみ(Baseline)」vs「因果特徴量あり(Proposed)」の比較を行い、
#       時系列ダイナミクス(Frequency Shift)に対する検知能力の差を実証する。
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

# srcフォルダへのパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.sentinel import CognitiveFeatureEngineer, AdversarialSimulator

def run_experiment():
    print("🚀 [Experiment] Starting Ablation Study: Causal Contribution Analysis...")

    # --------------------------------------------------------------------------
    # 1. Data Generation (Simulation)
    # --------------------------------------------------------------------------
    print("   -> Generating Simulation Data (Physics Domain / Frequency Attack)...")
    N = 6000
    t = np.linspace(0, 80, N)

    # 正常データ: ゆったりしたサイン波
    X_normal_source = pd.DataFrame({'Sensor_A': np.sin(t) + np.random.normal(0, 0.05, N)})
    y_normal = np.zeros(N)

    # 攻撃データ: 値の範囲は同じだが、周波数が高い（Frequency Shift）
    simulator = AdversarialSimulator()
    X_attack = simulator.inject_attack(X_normal_source.copy(), 'phys')
    y_attack = np.ones(N)

    # 攻撃期間のみを抽出（シミュレータの仕様に合わせる）
    attack_start = N // 2
    X_attack = X_attack.iloc[attack_start:]
    y_attack = y_attack[attack_start:]
    X_normal = X_normal_source.iloc[:attack_start]
    y_normal = y_normal[:attack_start]

    # データ結合
    X_all = pd.concat([X_normal, X_attack]).reset_index(drop=True)
    y_all = np.concatenate([y_normal, y_attack])

    # --------------------------------------------------------------------------
    # 2. Feature Engineering (The Critical Step)
    # --------------------------------------------------------------------------
    # 【重要】時系列順序が保たれている分割前に特徴量（速度・振動）を計算する
    print("   -> Calculating Causal Features (Before Shuffling)...")
    engineer = CognitiveFeatureEngineer()
    X_all_eng = engineer.process(X_all, 'phys')

    # --------------------------------------------------------------------------
    # 3. Train/Test Split
    # --------------------------------------------------------------------------
    X_train_eng, X_test_eng, y_train, y_test = train_test_split(
        X_all_eng, y_all, test_size=0.3, shuffle=True, random_state=42
    )

    # Baseline用データ作成（因果特徴量 _lag1, _roll_std を削除し、生データのみにする）
    raw_cols = [c for c in X_train_eng.columns if '_lag' not in c and '_roll' not in c and '_dev' not in c]
    X_train_raw = X_train_eng[raw_cols]
    X_test_raw = X_test_eng[raw_cols]

    # --------------------------------------------------------------------------
    # 4. Model Training & Evaluation
    # --------------------------------------------------------------------------
    
    # Model A: Baseline (Raw Features)
    print("\n   🥊 Round 1: Training Baseline Model (Raw Features)...")
    clf_A = LGBMClassifier(random_state=42, verbose=-1)
    clf_A.fit(X_train_raw, y_train)
    preds_A = clf_A.predict(X_test_raw)
    f1_A = f1_score(y_test, preds_A)
    print(f"      [Result] Baseline F1-Score: {f1_A:.4f}")

    # Model B: Proposed (Causal Invariants)
    print("\n   🥊 Round 2: Training Proposed Model (Causal Invariants)...")
    clf_B = LGBMClassifier(random_state=42, verbose=-1)
    clf_B.fit(X_train_eng, y_train)
    preds_B = clf_B.predict(X_test_eng)
    f1_B = f1_score(y_test, preds_B)
    print(f"      [Result] Proposed F1-Score: {f1_B:.4f}")

    # --------------------------------------------------------------------------
    # 5. Summary & Visualization
    # --------------------------------------------------------------------------
    improvement = f1_B - f1_A
    print(f"\n   🏆 Final Improvement: {improvement:+.4f}")
    
    if improvement > 0.2:
        print("      ✅ SUCCESS: Causal Invariants significantly improved detection.")
    else:
        print("      ⚠️ WARNING: Improvement is small. Check feature engineering.")

    # 簡易グラフ保存 (論文のFig 3用)
    results = pd.DataFrame({
        'Model': ['Baseline (Raw)', 'Proposed (Causal)'],
        'F1-Score': [f1_A, f1_B]
    })
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Model', y='F1-Score', data=results, palette='viridis')
    plt.title("Ablation Study: Impact of Causal Invariants", fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for index, row in results.iterrows():
        plt.text(index, row['F1-Score'] + 0.02, f"{row['F1-Score']:.3f}", color='black', ha="center")
    
    save_path = "experiments/result_ablation.png"
    plt.savefig(save_path)
    print(f"      📊 Graph saved to: {save_path}")

if __name__ == "__main__":
    run_experiment()
