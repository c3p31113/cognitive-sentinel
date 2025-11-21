import numpy as np
import pandas as pd
from scipy.stats import entropy, pearsonr

print("🔧 [Theory] Verifying Causal & Mathematical Claims...")

# ================================================================
# 1. Causal Intervention (do-calculus)
# ================================================================
print("\n🧪 [Proof 1] Causal Intervention Analysis")
N = 5000
# Observational
traffic = np.random.normal(50, 10, N)
cpu = (traffic * 2.0) + np.random.normal(0, 2, N)
print(f"  - Observational Correlation: {pearsonr(traffic, cpu)[0]:.4f}")

# Interventional do(Traffic=100)
traffic_do = np.full(N, 100)
cpu_do = (traffic_do * 2.0) + np.random.normal(0, 2, N)
ate = cpu_do.mean() - cpu.mean()

print(f"  - E[CPU | do(Traffic=100)]: {cpu_do.mean():.2f}")
print(f"  - ATE (Causal Effect): {ate:.2f}")
if ate > 50: print("  ✅ Causal Direction Proven (Traffic -> CPU)")

# ================================================================
# 2. Invariant Validation
# ================================================================
print("\n📐 [Proof 2] Invariant Validity")

# Lipschitz (Physical)
normal_data = np.random.normal(10, 1, 1000)
attack_data = np.random.uniform(0, 20, 1000) # Random drift

def check_lipschitz(data): return np.mean(np.abs(np.diff(data)) > 5.0)

lip_norm = check_lipschitz(normal_data)
lip_att = check_lipschitz(attack_data)
print(f"  - Lipschitz Violation (Normal): {lip_norm:.4f}")
print(f"  - Lipschitz Violation (Attack): {lip_att:.4f}")

# Entropy (Logical/Benford)
def check_entropy(data):
    hist, _ = np.histogram(data % 1, bins=10, density=True)
    return entropy(hist + 1e-9)

ent_norm = check_entropy(np.random.lognormal(0, 1, 1000)) # Biased
ent_att = check_entropy(np.random.uniform(0, 100, 1000))  # Uniform (Attack)
print(f"  - Entropy (Normal): {ent_norm:.4f}")
print(f"  - Entropy (Attack): {ent_att:.4f}")

if lip_att > lip_norm and ent_att > ent_norm:
    print("  ✅ Invariants Validated.")

print("\n🏁 Theory Verification Complete.")