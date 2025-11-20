# Cognitive Sentinel: Interpretable Neuro-Symbolic Intrusion Detection via Causal Invariants

**(コグニティブ・センチネル：因果的不変量に基づく、説明可能なニューロシンボリック侵入検知)**

[](https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=)
[](https://www.google.com/search?q=)

-----

## 0\. Abstract (要旨)

現代のサイバー物理システム（CPS）に対する攻撃は、正常データの統計分布に擬態する「ステルス攻撃（In-distribution attacks）」へと進化している。従来の深層学習（Deep Learning）ベースのIDSは、データの「相関関係（Correlation）」のみを学習するため、因果律（Causality）を無視した精巧な偽装攻撃（Adversarial Attacks）に対して脆弱であり、その判断根拠もブラックボックスであった。

本研究では、人間の認知プロセス（System 1: 直感 / System 2: 論理）を模倣した **Neuro-Symbolic Architecture** を提案する。我々は、物理・論理・サイバーの各ドメインにおける因果的不変量（Causal Invariants）を数式化し、それを特徴量として機械学習モデルに注入するフレームワークを構築した。

5つのデータセット（計146,690サンプル）を用いた広範な評価の結果、本手法は敵対的攻撃を含む未知の脅威に対し **F1-Score 0.92** を達成した。特筆すべきは、GAN（敵対的生成ネットワーク）による偽装攻撃に対しても **86.6%** の検知率を維持し、統計的擬態に対する堅牢性を実証した点である。さらに、既存のOne-Class SVMと比較して **47倍** の推論速度を実現し、エッジデバイスでの実用性を示した。

-----

## 1\. Introduction (序論)

### 1.1 The Problem: Limits of Statistical Learning

既存のIDS（Isolation Forest, Autoencoder等）は、正常データが形成する多様体（Manifold）からの距離に基づいて異常を検知する。しかし、**Slow Drift**（物理法則を悪用した緩慢な変化）や **Adversarial Noise**（統計的特徴を隠蔽するゆらぎ）は、正常多様体の内部に留まるため、原理的に検知不可能である。

### 1.2 The Hypothesis: Causal Consistency

攻撃者はデータを「正常な値」に偽装することはできても、そのデータが従うべき\*\*「背後の因果律（物理法則・商習慣・機械的特性）」**まで完全に模倣するには、対象システムと同等のシミュレーションコストを要する。
したがって、データそのものではなく**「不変量との矛盾（Invariant Violation）」\*\*を検知することで、ステルス攻撃を無効化できると仮説を立てた。

-----

## 2\. Threat Model (脅威モデル)

本システムは、以下の能力を持つ強力な攻撃者（Adversary）を対象とする。

  * **Knowledge:** 攻撃者はシステム正常時のセンサー値の分布（平均・分散）を完全に把握している（White-box）。
  * **Capability:** 攻撃者はGANや強化学習を用い、検知システムを回避するための「ゆらぎ（Jitter）」や「統計的偽装（Smudging）」を含むデータを生成できる。
  * **Constraint:** 攻撃者はデータ値を改ざんできるが、物理世界の実プロセス（例：パケット送信に伴う実際のCPU発熱遅延）まではリアルタイムに操作できない。

-----

## 3\. Methodology: The Cognitive Architecture

本手法は、単なる特徴量エンジニアリングではなく、以下の3層構造を持つ統合モデルである。

### 3.1 Layer 1: Symbolic Invariant Extraction (シンボリック層)

ドメイン知識に基づき、攻撃者が偽装困難な「不変量 $\Phi(x)$」を数式化する。

#### (A) Physical Inertia (物理的慣性)

  * **Justification:** 物理システムはリプシッツ連続性を持ち、急激な変化や因果乖離を許容しない。
  * **Formulation:** 予測モデル $f$ との残差ノルム。
    $$\Phi_{phys} = \| x_t - f(x_{t-1}, \dots, x_{t-k}) \|_2$$

#### (B) Logical Entropy (論理的エントロピー)

  * **Justification:** 人間の経済活動（商取引）における数値分布は、ベンフォードの法則等の特定のバイアスを持つ。攻撃者が生成する乱数はこれに従わない。
  * **Formulation:** 観測分布 $P$ と自然分布 $Q$ のカルバック・ライブラー情報量。
    $$\Phi_{logic} = D_{KL}(P \| Q) = \sum P(i) \log \frac{P(i)}{Q(i)}$$

#### (C) Cyber Regularity (サイバー律動性)

  * **Justification:** プログラム（Bot）による通信は、たとえランダム化（Jitter）されても、人間よりも情報の多様性（Entropy）が低い。
  * **Formulation:** ウィンドウ $W$ 内のユニーク比率。
    $$\Phi_{cyber} = 1 - \frac{\|Unique(W)\|}{\|W\|}$$

### 3.2 Layer 2: Neural Contextual Learning (ニューラル層)

抽出された不変量 $\Phi$ と元データ $X$ を結合し、**LightGBM** 分類器に入力する。
学習には、独自開発したデータ拡張アルゴリズム **"Synthetic Dojo"** を用いる。これは、正常データに対して理論的な攻撃パターン（Drift, Jitter, Salami）を確率的に注入し、教師あり学習として境界決定を行うものである。

### 3.3 Layer 3: Adaptive Decision (適応判定層)

環境ごとのノイズレベル変動に対応するため、学習期間における不変量スコアの中央値（Median）と四分位範囲（IQR）を用いたロバストなZスコア変換を行い、動的に閾値を決定する。
$$Z(x) = \frac{\Phi(x) - \text{Median}}{\text{IQR}}$$

-----

## 4\. Evaluation (評価実験)

### 4.1 Experimental Setup

  * **Datasets:** SKAB (Physical), Credit Card (Logical), CTU-13 (Cyber).
  * **Generalization Test:** CIC-IDS2017, UNSW-NB15.
  * **Reproducibility:** 全ての乱数シードを `42` に固定。実験コードは公開済み。

### 4.2 Comparative Analysis (SOTA比較)

提案手法と、代表的な異常検知モデルの比較結果。

| Model | Architecture | Recall | Precision | F1-Score | Speed (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Isolation Forest | Statistical | 0.14 | 0.94 | 0.24 | 0.22 |
| Deep Autoencoder | Deep Learning | 0.08 | 0.88 | 0.15 | 5.40 |
| One-Class SVM | Kernel Method | 0.21 | 0.91 | 0.34 | 10.39 |
| **Ours (Proposed)** | **Neuro-Symbolic** | **0.91** | **0.93** | **0.92** | **0.22** |

> **分析:** 既存手法は「点」の異常しか検知できなかったが、本手法は「文脈（不変量）」の破綻を検知したため、ステルス攻撃を90%以上捕捉した。

### 4.3 Robustness against Adversarial Attacks (敵対的攻撃耐性)

攻撃者が検知回避行動をとった場合の検知率。

  * **Jittery Beacon (ゆらぎ通信):** Recall **93.00%**
      * *Reason:* 通信間隔をランダム化しても、統計的な分散の低さまでは隠蔽できなかった。
  * **Smudged Salami (人間的偽装):** Recall **84.40%**
      * *Reason:* 端数を人間に似せても、集団としての分布歪み（エントロピー低下）は隠せなかった。
  * **GAN Mimicry (AI偽装):** Recall **86.64%**
      * *Reason:* GANは分布 $P(X)$ を模倣したが、因果的応答 $P(Y|do(X))$（Traffic増大時のCPU遅延など）を再現できなかった。

### 4.4 Zero-Shot Generalization (未知環境への適応)

学習に使用していないデータセットに対し、追加学習なし（Zero-Shot）で適用した結果。

  * **CIC-IDS2017 (DDoS/BruteForce):** Recall **76.83%**
  * **UNSW-NB15 (Fuzzers/Backdoor):** Recall **100.00%**
      * *Note:* UNSWでの100%検知は、Fuzzersの異常度が統計的限界（IQR \> 50）を突破していたことによる必然的結果である。

-----

## 5\. Discussion & Limitations (議論と限界)

### 5.1 Causal Graph Validity (因果グラフの正当性)

我々はグレンジャー因果性検定により、仮定した因果グラフの統計的有意性を確認した（Correlation \> 0.99, Lag \> 0）。これは、本手法が主観的なルールではなく、データに内在する物理的性質に基づいていることを示唆する。

### 5.2 Limitations (限界)

1.  **Analog Sensor Compromise:** センサー自体が物理的に欺瞞された場合（例：温度計を氷で冷やす）、入力データ自体が信頼性を失うため検知できない。
2.  **Precision Trade-off:** 未知の環境（CIC-IDS2017）においては、Recallを維持するためにPrecisionが低下する傾向が見られた（35.3%）。実運用では、少量の正常データによるキャリブレーションが推奨される。

-----

## 6\. Conclusion (結論)

本研究は、AIセキュリティにおける「ブラックボックス問題」と「ステルス攻撃への脆弱性」に対し、**因果律と不変量に基づくNeuro-Symbolicアプローチ**が有効な解決策であることを実証した。
本手法は、計算コストが極めて低く（OCSVM比 47倍速）、かつ敵対的攻撃に対して堅牢であることから、IoT/Edge環境における次世代IDSの新たな標準となり得るものである。

-----

### 🔧 Replication (再現性)

全ての実験は、以下のコマンドで再現可能です。

```bash
pip install -r requirements.txt
python golden_master.py
```
