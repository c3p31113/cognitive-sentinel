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
    def process(self, df, domain):
        df_eng = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if domain == 'phys':
            # Physics: Inertia (Speed) & Volatility (Vibration)
            base_cols = [c for c in numeric_cols if 'lag' not in c and 'roll' not in c and 'anomaly' not in c]
            for c in base_cols:
                # 1. Inertia/Velocity
                df_eng[f'{c}_lag1'] = df[c].diff().fillna(0)
                # 2. Volatility/Dynamics
                df_eng[f'{c}_roll_std'] = df[c].rolling(window=10).std().fillna(0)
                # 3. Deviation
                roll_mean = df[c].rolling(window=10).mean().fillna(0)
                df_eng[f'{c}_dev'] = df[c] - roll_mean

        elif domain == 'logic':
            if 'Amount' in df.columns:
                df_eng['Amt_dec'] = df['Amount'] % 1
                df_eng['Amt_dist_int'] = np.abs(0.5 - (df['Amount'] % 1))
                df_eng['Amt_log'] = np.log1p(df['Amount'])

        elif domain == 'cyber':
            targets = ['Total Length of Fwd Packets', 'Flow Duration']
            for c in targets:
                if c in df.columns:
                    window_size = 20
                    def nunique_ratio(x): return len(np.unique(x)) / len(x)
                    df_eng[f'{c}_uniq20'] = df[c].rolling(window=window_size).apply(nunique_ratio, raw=True).fillna(1)
                    df_eng[f'{c}_std20'] = df[c].rolling(window=window_size).std().fillna(0)

        return df_eng.fillna(0)

# ==============================================================================
# [Module 2] The Synthetic Dojo (Data Augmentation)
# ==============================================================================
class AdversarialSimulator:
    """
    Generates synthetic stealth attacks for training.
    CRITICAL FIX: Attacks must be 'in-distribution' (same value range) to prove
    the value of causal invariants. Simple addition makes it too easy for baseline.
    """
    def _strip(self, df):
        return [c for c in df.columns if '_dev' not in c and '_lag' not in c and 
                '_dec' not in c and '_dist' not in c and '_uniq' not in c and 
                '_roll' not in c and '_std' not in c and '_log' not in c]
    
    def inject_attack(self, df, domain):
        base_cols = self._strip(df)
        inj = df[base_cols].copy()
        N = len(inj)
        
        if domain == 'phys':
            t = np.linspace(0, 100, N)
            
            for c in inj.columns:
                # 既存の振幅(強さ)を取得
                amp = np.std(inj[c]) if np.std(inj[c]) > 0 else 1.0
                mean_val = np.mean(inj[c])
                
                # 半分はFrequency、半分はFreeze
                if np.random.rand() > 0.5:
                    # 【修正】足し算(+=)ではなく、上書き(=)にする
                    # これにより値の範囲が正常と同じになり、Baselineを欺ける
                    freq_attack = np.sin(t * 8.0) * amp + mean_val
                    inj[c] = freq_attack
                    
                    # 難易度調整のノイズ
                    inj[c] += np.random.normal(0, 0.05 * amp, N)
                else:
                    # Freeze Attack
                    inj[c] = mean_val + np.random.normal(0, 0.001, N)
                
        elif domain == 'logic':
            if 'Amount' in inj.columns: 
                inj['Amount'] += np.random.uniform(0.01, 0.99, size=len(inj))
                
        elif domain == 'cyber':
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
        self.clf = LGBMClassifier(n_estimators=200, verbose=-1, random_state=42)
        self.reg = None
        self.iso = None
        self.qt = QuantileTransformer(output_distribution='normal', random_state=42)
        
        if domain == 'logic': self.scaler = MinMaxScaler()
        else: self.scaler = RobustScaler()
        
        self.features_ = None
        self.threshold_ = 0.5 

    def fit(self, X, y):
        # 1. Feature Engineering (Normal Data)
        X_eng = self.engineer.process(X, self.domain)
        self.features_ = X_eng.columns
        
        # 2. Synthetic Dojo
        X_norm_raw = X[y==0].copy()
        
        try:
            # Generate Attack Raw Data
            X_synth_raw = self.simulator.inject_attack(X_norm_raw, self.domain)
            # Extract Features
            X_synth_eng = self.engineer.process(X_synth_raw, self.domain)
            X_synth_eng = X_synth_eng.reindex(columns=X_eng.columns, fill_value=0)
            
            # Combine
            y_synth = pd.Series(np.ones(len(X_synth_eng)))
            
            # Pandas Series check
            if not isinstance(y, pd.Series): y = pd.Series(y)
                
            X_combined = pd.concat([X_eng, X_synth_eng], ignore_index=True)
            y_combined = pd.concat([y, y_synth], ignore_index=True)
            
        except Exception as e:
            print(f"Warning: Dojo generation failed ({e}). Training on provided data only.")
            X_combined = X_eng
            y_combined = y
            
        # 3. Train System 1
        X_s = self.scaler.fit_transform(X_combined)
        self.clf.fit(X_s, y_combined)
        
        # 4. Train System 2
        X_norm_s = self.scaler.transform(X_eng[y==0])
        
        if self.domain == 'phys':
            if X_norm_s.shape[1] > 1:
                self.reg = LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
                self.reg.fit(np.delete(X_norm_s, 0, axis=1), X_norm_s[:, 0])
                preds = self.reg.predict(np.delete(X_norm_s, 0, axis=1))
                errs = np.abs(X_norm_s[:, 0] - preds)
                self.qt.fit(errs.reshape(-1, 1))
        else:
            cont = 0.05 if self.domain == 'logic' else 0.1
            self.iso = IsolationForest(contamination=cont, random_state=42, n_jobs=-1)
            self.iso.fit(X_norm_s)
            errs = 0.5 - self.iso.decision_function(X_norm_s)
            self.qt.fit(errs.reshape(-1, 1))

    def predict(self, X):
        X_eng = self.engineer.process(X, self.domain)
        X_eng = X_eng.reindex(columns=self.features_, fill_value=0)
        X_s = self.scaler.transform(X_eng)
        
        # System 1
        s_clf = self.clf.predict_proba(X_s)[:, 1]
        
        # System 2
        if self.domain == 'phys' and self.reg is not None:
            pred = self.reg.predict(np.delete(X_s, 0, axis=1))
            raw_err = np.abs(X_s[:, 0] - pred)
        elif self.iso is not None:
            raw_err = 0.5 - self.iso.decision_function(X_s)
        else:
            raw_err = np.zeros(len(X))
            
        try:
            s_sym = self.qt.transform(raw_err.reshape(-1, 1)).flatten()
            s_sym = np.clip(s_sym, 0, 10)
        except:
            s_sym = np.zeros(len(X))
        
        final_score = (s_clf * 0.8) + (s_sym * 0.2)
        return (final_score > self.threshold_).astype(int)
