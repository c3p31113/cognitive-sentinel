import numpy as np
import pandas as pd
import warnings
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer

warnings.filterwarnings("ignore")

# ==============================================================================
# [Module 1] The Prefrontal Cortex (Feature Engineering / System 2)
# ==============================================================================
class CognitiveFeatureEngineer:
    """
    System 2: Extracts Causal Invariants from raw data.
    CRITICAL: This must be applied BEFORE any data shuffling to preserve temporal dynamics.
    """
    def process(self, df, domain):
        df_eng = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if domain == 'phys':
            # Physics: Inertia (Speed) & Volatility (Vibration)
            # Ablation Studyで証明された重要特徴量: diff(速度) と rolling_std(振動)
            base_cols = [c for c in numeric_cols if 'lag' not in c and 'roll' not in c and 'anomaly' not in c]
            for c in base_cols:
                # 1. Inertia/Velocity: 物理量は急に値が飛ばない (Lipschitz)
                df_eng[f'{c}_lag1'] = df[c].diff().fillna(0)
                
                # 2. Volatility/Dynamics: 物理量は常に揺らいでいる (Dynamic Consistency)
                # Freeze攻撃(std=0)やFrequency攻撃(std高)を検知
                df_eng[f'{c}_roll_std'] = df[c].rolling(window=10).std().fillna(0)
                
                # 3. Deviation: 予測からの乖離
                roll_mean = df[c].rolling(window=10).mean().fillna(0)
                df_eng[f'{c}_dev'] = df[c] - roll_mean

        elif domain == 'logic':
            # Logic: Statistical Laws (Benford/Salami)
            if 'Amount' in df.columns:
                df_eng['Amt_dec'] = df['Amount'] % 1
                df_eng['Amt_dist_int'] = np.abs(0.5 - (df['Amount'] % 1))
                df_eng['Amt_log'] = np.log1p(df['Amount'])

        elif domain == 'cyber':
            # Cyber: Mechanical Regularity (Beacon)
            targets = ['Total Length of Fwd Packets', 'Flow Duration']
            for c in targets:
                if c in df.columns:
                    window_size = 20
                    def nunique_ratio(x): return len(np.unique(x)) / len(x)
                    # Note: rolling apply is slow, but necessary for entropy calculation
                    df_eng[f'{c}_uniq20'] = df[c].rolling(window=window_size).apply(nunique_ratio, raw=True).fillna(1)
                    df_eng[f'{c}_std20'] = df[c].rolling(window=window_size).std().fillna(0)

        return df_eng.fillna(0)

# ==============================================================================
# [Module 2] The Synthetic Dojo (Data Augmentation)
# ==============================================================================
class AdversarialSimulator:
    """
    Generates synthetic stealth attacks for training (Supervised Training).
    This teaches the model 'what a violation of physics looks like'.
    """
    def _strip(self, df):
        # Remove engineered features to get back to raw state for injection
        return [c for c in df.columns if '_dev' not in c and '_lag' not in c and 
                '_dec' not in c and '_dist' not in c and '_uniq' not in c and 
                '_roll' not in c and '_std' not in c and '_log' not in c]
    
    def inject_attack(self, df, domain):
        base_cols = self._strip(df)
        inj = df[base_cols].copy()
        N = len(inj)
        
        if domain == 'phys':
            # 1. Frequency Attack (High Vibration) - from Ablation Study
            t = np.linspace(0, 100, N)
            high_freq_noise = np.sin(t * 8.0) * 0.5 # High frequency oscillation
            
            # 2. Freeze Attack (Zero Dynamics)
            freeze_val = inj.mean() # Fixed value
            
            # Randomly apply either Freeze or Frequency attack
            for c in inj.columns:
                if np.random.rand() > 0.5:
                    inj[c] += high_freq_noise # Violates Inertia
                else:
                    inj[c] = freeze_val[c] + np.random.normal(0, 0.001, N) # Violates Dynamics (almost zero std)
                
        elif domain == 'logic':
            # Smudged Salami
            if 'Amount' in inj.columns: 
                inj['Amount'] += np.random.uniform(0.01, 0.99, size=len(inj))
                
        elif domain == 'cyber':
            # Jittery Beacon (Regularity Violation)
            for i in range(0, len(inj), 10):
                if 'Total Length of Fwd Packets' in inj.columns: 
                    inj.iloc[i, inj.columns.get_loc('Total Length of Fwd Packets')] = 77.0
                if 'Flow Duration' in inj.columns: 
                    inj.iloc[i, inj.columns.get_loc('Flow Duration')] = 123.0
                    
        return inj

# ==============================================================================
# [Module 3] The Cognitive Sentinel (Main Class)
# ==============================================================================
class CognitiveSentinel:
    def __init__(self, domain='cyber'):
        self.domain = domain
        self.engineer = CognitiveFeatureEngineer()
        self.simulator = AdversarialSimulator()
        
        # System 1: Causal-Informed Classifier
        self.clf = LGBMClassifier(n_estimators=200, verbose=-1, random_state=42)
        
        # System 2: Anomaly Scorers (Regressor / IsolationForest)
        self.reg = None
        self.iso = None
        self.qt = QuantileTransformer(output_distribution='normal', random_state=42)
        if domain == 'logic': self.scaler = MinMaxScaler()
        else: self.scaler = RobustScaler()
        
        self.features_ = None
        self.threshold_ = 0.8 # Default threshold

    def fit(self, X, y):
        """
        Training Pipeline:
        1. Feature Engineering (on Raw Data) -> Preserves Causal Dynamics
        2. Data Augmentation (Synthetic Dojo) -> Teaches Violations
        3. System 1 & 2 Training
        """
        # 1. Feature Engineering (Normal Data)
        X_eng = self.engineer.process(X, self.domain)
        self.features_ = X_eng.columns
        
        # 2. Synthetic Dojo (Augmentation)
        # We generate attacks based on raw normal data, THEN engineer features.
        # This ensures the 'attack features' (like high velocity) are correctly calculated.
        X_raw_only = X[y==0].copy() # Get only normal samples to corrupt
        
        try:
            # Generate Attack Raw Data
            X_synth_raw = self.simulator.inject_attack(X_raw_only, self.domain)
            # Extract Causal Features from Attack Data
            X_synth_eng = self.engineer.process(X_synth_raw, self.domain)
            X_synth_eng = X_synth_eng.reindex(columns=X_eng.columns, fill_value=0)
            
            # Combine Normal + Synthetic Attack
            y_synth = pd.Series(np.ones(len(X_synth_eng)))
            X_combined = pd.concat([X_eng, X_synth_eng])
            y_combined = pd.concat([y, y_synth])
            
        except Exception as e:
            print(f"Warning: Dojo generation failed ({e}). Training on provided data only.")
            X_combined = X_eng
            y_combined = y
            
        # 3. Train System 1 (Causal-Informed Classifier)
        X_s = self.scaler.fit_transform(X_combined)
        self.clf.fit(X_s, y_combined)
        
        # 4. Train System 2 (Unsupervised Baseline for Hybrid Scoring)
        X_norm_s = self.scaler.transform(X_eng[y==0])
        
        if self.domain == 'phys':
            # Regressor for Physics (Predict next state)
            # Predict X[t] based on X[t-1]... (Simplified for demo)
            self.reg = LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
            # Train to map features to target (Self-Supervised)
            self.reg.fit(np.delete(X_norm_s, 0, axis=1), X_norm_s[:, 0])
            
            preds = self.reg.predict(np.delete(X_norm_s, 0, axis=1))
            errs = np.abs(X_norm_s[:, 0] - preds)
            self.qt.fit(errs.reshape(-1, 1))
            
        else:
            # IsoForest for Logic/Cyber
            cont = 0.05 if self.domain == 'logic' else 0.1
            self.iso = IsolationForest(contamination=cont, random_state=42, n_jobs=-1)
            self.iso.fit(X_norm_s)
            errs = 0.5 - self.iso.decision_function(X_norm_s)
            self.qt.fit(errs.reshape(-1, 1))

    def predict(self, X):
        # 1. Feature Engineering
        X_eng = self.engineer.process(X, self.domain)
        X_eng = X_eng.reindex(columns=self.features_, fill_value=0)
        X_s = self.scaler.transform(X_eng)
        
        # 2. System 1 Prediction (Supervised)
        s_clf = self.clf.predict_proba(X_s)[:, 1]
        
        # 3. System 2 Prediction (Unsupervised / Residual)
        if self.domain == 'phys':
            pred = self.reg.predict(np.delete(X_s, 0, axis=1))
            raw_err = np.abs(X_s[:, 0] - pred)
        else:
            raw_err = 0.5 - self.iso.decision_function(X_s)
            
        s_sym = self.qt.transform(raw_err.reshape(-1, 1)).flatten()
        s_sym = np.clip(s_sym, 0, 10)
        
        # 4. Hybrid Score Fusion
        # System 1 is weighted higher due to Causal Dojo training
        final_score = (s_clf * 0.8) + (s_sym * 0.2) 
        
        return (final_score > 0.5).astype(int) # Simple threshold for demo