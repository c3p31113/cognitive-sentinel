import sys
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, precision_score

# Import src module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.sentinel import CognitiveSentinel, AdversarialSimulator

def load_data():
    data = {}
    print("📂 Loading Data...")
    # Mock Loading for demonstration (Replace with real CSV loading logic)
    # In real usage, put your CSV loading code here from previous notebooks
    # For now, we use simulation to ensure it runs out-of-the-box
    return data

def main():
    print("🚀 [Experiment] Running Benchmark Suite...")
    
    # --- Simulation for Reproducibility (Replace with Real Data Load) ---
    # This ensures the script runs immediately for reviewers
    from src.sentinel import AdversarialSimulator
    injector = AdversarialSimulator()
    
    # 1. Physics (SKAB Simulation)
    N = 5000
    t = np.linspace(0, 50, N)
    df_phys_train = pd.DataFrame({'Sensor': np.sin(t) + np.random.normal(0, 0.1, N)})
    y_phys_train = np.zeros(N)
    
    # Attack: Drift
    df_phys_test = injector.inject_attack(df_phys_train, 'phys')
    y_phys_test = np.ones(N)
    
    # 2. Run Sentinel
    print("\n🔬 Domain: Physical (SKAB)")
    sentinel = CognitiveSentinel('phys')
    sentinel.fit(df_phys_train, y_phys_train)
    preds = sentinel.predict(df_phys_test)
    
    print(classification_report(y_phys_test, preds))
    print(f"Recall: {recall_score(y_phys_test, preds)*100:.2f}%")
    
    # 3. Cyber (CTU-13 Simulation)
    print("\n🔬 Domain: Cyber (CTU-13)")
    df_cyber_train = pd.DataFrame({
        'Total Length of Fwd Packets': np.random.lognormal(3, 1, N),
        'Flow Duration': np.random.exponential(100, N)
    })
    y_cyber_train = np.zeros(N)
    
    # Attack: Beacon
    df_cyber_test = injector.inject_attack(df_cyber_train, 'cyber')
    y_cyber_test = np.ones(N)
    
    sentinel = CognitiveSentinel('cyber')
    sentinel.fit(df_cyber_train, y_cyber_train)
    preds = sentinel.predict(df_cyber_test)
    
    print(classification_report(y_cyber_test, preds))
    print(f"Recall: {recall_score(y_cyber_test, preds)*100:.2f}%")
    
    print("\n✅ Benchmark Complete.")

if __name__ == "__main__":
    main()