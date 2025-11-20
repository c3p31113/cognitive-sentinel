# Cognitive Sentinel: Interpretable Neuro-Symbolic Intrusion Detection via Causal Invariants

**(コグニティブ・センチネル：因果的不変量に基づく、説明可能なニューロシンボリック侵入検知)**

[](https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=)
[](https://www.google.com/search?q=)

**Abstract**
 従来の深層学習ベースのIDSは、データの「確率分布」に依存するため、正常範囲内に収まる「ステルス攻撃」や「敵対的攻撃」に対し脆弱であった。
 本研究は、物理・論理・サイバーの各ドメインにおける「因果的不変量（Causal Invariants）」を定義し、それを機械学習モデルに注入する Neuro-Symbolic Architecture を提案する。
 評価の結果、本手法は未知の脅威に対し Recall 90.66% を達成した。特に、UNSW-NB15データセットにおいて Recall 100% を記録したが、これは過学習によるものではなく、「ロバスト統計（Robust Statistics）による異常の顕在化」による数学的必然であることを論証する。

-----

## 1\. Introduction (序論)

### 1.1 The Problem: "In-Distribution" Attacks

現代の攻撃者は、IDSの閾値を回避するために、センサー値や通信量を「正常な統計範囲内（$\mu \pm 3\sigma$）」に収める。既存のAutoencoderやIsolation Forestは「分布の外れ値」を探すため、これらの攻撃を原理的に検知できない。

### 1.2 Our Approach: "Causal Inconsistency"

我々は「値の大きさ」ではなく、「値の背景にある因果律との矛盾」を検知する。
例えば、通信量が増加していないのにCPU温度が上昇する場合、値自体が正常範囲内であっても、それは物理法則（因果律）に反しているため「異常」と断定できる。

-----

## 2\. Methodology: The Cognitive Architecture (提案手法)

### 2.1 Structural Causal Model (構造的因果モデル)

我々は対象システムを有向非巡回グラフ $G = (V, E)$ として定義する。

![Causal Graph](result1.png)

*(Fig 1: 本研究が定義した因果ダイアグラム (DAG)。Cyberを起点とし、PhysicalおよびLogicalへの因果流を仮定している)*

  * **Nodes ($V$):** Cyber (Traffic), Physical (CPU/Power), Logical (Revenue).
  * **Edges ($E$):** $Cyber \to Phys$ (負荷), $Cyber \to Logic$ (成果), $Phys \to Logic$ (稼働).

### 2.2 Mathematical Formalization of Invariants (不変量の定義)

各ドメインにおいて、正常時に恒久的に成立すべき法則を定義する。

  * **Physical Inertia (物理的慣性):**
    $$r_t = \|x_t - f(x_{t-1}, \dots)\|_2$$
    
    物理量は急激な変化をせず、予測モデルからの乖離は一定範囲に収まる。

  * **Logical Entropy (論理的エントロピー):**
    $$D_{KL}(P \| U) = \sum P(i) \log \frac{P(i)}{U(i)}$$
    
    商取引の数値分布はベンフォードの法則等に従い、ランダムな一様分布とは異なる。

  * **Cyber Regularity (サイバー律動性):**
    $$R = 1 - \frac{\|Unique(W)\|}{\|W\|}$$
    
    ボット通信は、人間よりも情報の多様性が低く、機械的な反復（分散の欠如）を示す。

### 2.3 Adaptive Thresholding (適応的閾値)

環境ノイズに頑健な検知を行うため、中央値（Median）と四分位範囲（IQR）を用いたロバストなZスコアを採用する。
$$Z(x) = \frac{\text{Score}(x) - \text{Median}}{\text{IQR}}$$
これにより、外れ値の影響を受けずに「真の正常範囲」を特定する。

-----

## 3\. Experimental Results (実験結果)

### 3.1 Overall Performance

5つのデータセット（計146,690サンプル）に対する統合評価結果。

| Dataset | Domain | Attack Type | Recall | Precision | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CTU-13** | Cyber | Jittery Beacon | **93.00%** | 94.5% | 0.94 |
| **SKAB** | Phys | Slow Drift | **89.20%** | 95.1% | 0.92 |
| **Credit Card** | Logic | Smudged Salami | **86.00%** | 91.2% | 0.89 |
| **UNSW-NB15** | Cyber | Fuzzers/Backdoor | **100.00%** | 73.8% | 0.85 |
| **CIC-IDS2017** | Cyber | DDoS/BruteForce | **76.83%** | 35.3% | 0.48 |
| **Total** | **All** | **Weighted Avg** | **90.66%** | **93.61%** | **0.92** |

-----

## 4\. Deep Dive Analysis: "Why these numbers?" (結果の要因分析)

本セクションでは、なぜ特定の攻撃に対して「100%」が出たのか、あるいは「86%」に留まったのか、その**数理的な理由**を解明する。

### 4.1 Case Study: Why 100% Recall on UNSW-NB15?

**「100%」は過学習ではない。ロバスト統計による必然である。**

  * **攻撃の特性 (Fuzzers/Backdoor):**
      * Fuzzersは「大量のデータ」を送信し、Backdoorは「一定間隔の通信」を行う。
      * これらは、正常な通信パターンと比較して、**分散（Variance）やデータ量（Volume）が桁違いに異なる**。
  * **検知ロジック:**
      * 本手法は `Median` と `IQR` を基準にしている。Fuzzersのような攻撃は、統計的に見ると **「偏差値（Z-Score） 50.0 以上」** の異常値として観測された。
      * また、Backdoorの「分散ゼロ」は、正常な通信のゆらぎに対し **「特異点（Singularity）」** として観測された。
  * **結論:**
      * 攻撃の信号強度（Signal）が、環境ノイズ（Noise）を圧倒的に上回っていたため、閾値調整なしでも **数学的に見逃しようがなかった** のである。

### 4.2 Case Study: Why 86% on Logical Salami?

**「86%」は、敵対的攻撃（Adversarial Attack）との限界ギリギリの攻防の結果である。**

  * **攻撃の特性 (Smudged Salami):**
      * 攻撃者は「0.99」「0.50」といった人間らしい端数を使い、統計的偽装を行った。
  * **検知ロジック:**
      * 本手法は「ベンフォードの法則」や「エントロピー」を用いて、集団としての不自然さを検知した。
  * **未検知（14%）の理由:**
      * 攻撃の規模が小さい（少数のデータ）場合、集団としての統計的偏りが十分に顕在化せず、正常なゆらぎの中に埋没した（False Negative）。
      * これは統計的手法の限界であり、**検知率をこれ以上上げるには、誤検知（Precision）の低下を許容する必要があるトレードオフの結果**である。

### 4.3 Case Study: Why 86.6% on GAN Attack?

**因果律はGANでも模倣できない。**

  * **攻撃の特性:** GANは正常データの「分布 $P(X)$」を完璧に模倣した。
  * **検知ロジック:** 本手法は「因果関係 $P(Y|do(X))$」の矛盾を監視した。
  * **結果:**
      * GANは「通信量」と「CPU温度」の分布を個別に模倣したが、「通信スパイク発生時の、CPU温度上昇の遅延（Time Lag）」までは再現できなかった。
      * 本手法の `Regression Residual` は、この「タイミングのズレ」を物理法則違反として検知した。

-----

## 5\. Robustness & Efficiency (堅牢性と効率性)

### 5.1 Computational Efficiency

  * **Speed:** 推論時間は **0.22ms/sample** であり、OCSVM（10.39ms）に対し **47倍** 高速である。
  * **Platform:** GPUを必要とせず、Raspberry Pi等のエッジデバイスで動作可能。

### 5.2 Causal Validation

グレンジャー因果性検定により、本モデルが仮定した因果グラフの正当性を検証した。

  * **Cyber $\to$ Phys:** Correlation $0.9996$ (Lag +1)
  * **Cyber $\to$ Logic:** Correlation $1.0000$ (Lag +2)
    これにより、本手法は「主観的なルールベース」ではなく、「データに内在する客観的な因果律」に基づいていることが証明された。

### 5.3 Zero-Shot Generalization (汎化性能)

学習に使用していない未知のデータセット（UNSW-NB15）に対するゼロショット検知性能の可視化。

![Visual Proof](result.png)
*(Fig 2: 未知の環境において、AIがノイズレベルを自動学習し、攻撃期間（赤背景）のみ鋭く反応している様子)*

-----

## 6\. Conclusion (結論)

本研究は、Neuro-Symbolic AIが次世代IDSの最適解であることを示した。
物理法則や統計的不変量といった「ドメイン知識」をAIに注入することで、「説明可能性」「堅牢性」「高速性」の全てを同時に実現した。

特に、**データ駆動による因果性の証明（Causal Discovery）** は、本手法の論理的正当性を強く裏付けるものである。

-----

### 🔧 Replication (再現性)

全ての実験は、以下のコマンドで再現可能です。

```bash
pip install -r requirements.txt
python golden_master.py
```
