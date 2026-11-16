**# BERT-word-embeddings-for-cross-project-story-point-estimation
BERT word embeddings for cross project story point estimation with temporal ordering. This is a NLP course project. Implemented preprocessing, BERT word embedding, Training multiple models using 5 fold expanding window training data split. Error calculation using MAE and MdAE
**# Problem Statement 4: BERT Word Embeddings for Cross-Project Software Story Point Estimation with Temporal Ordering

## Group Members

* **Kasireddy Koti Reddy** – \[B230373CS\]
* **** – \[Roll Number / ID\]
* **\[Member 3 Name\]** – \[Roll Number / ID\]
* **\[Member 4 Name\]** – \[Roll Number / ID\]

*(Update with your team details prior to final submission)*

## Problem Statement

> **"Evaluate BERT-based word embeddings for predicting story points of later issues in a target project using source-project training data and chronological ordering."**

Predicting Agile software story points across distinct software repositories presents three core challenges:

1. **Cross-Project Domain Shift:** Models trained on historical issues from a source project ($P_{\text{source}}$) must generalize to completely unseen issues from a disjoint target project ($P_{\text{target}}$, where $P_{\text{source}} \neq P_{\text{target}}$) across **39 assigned cross-project pairs**.
2. **Temporal / Chronological Ordering:** Real-world estimation requires predicting future tickets from historical backlogs without look-ahead bias ($\max(\text{Dev Date}) \le \min(\text{Test Date})$).
3. **Semantic Representation & Heavy-Tailed Targets:** Issue descriptions vary widely in length and technical vocabulary, while story point labels exhibit strong Fibonacci clustering ($1, 2, 3, 5, 8, 13, 21$) and heavy right-skew.

## Objectives

1. **BERT Word Embedding Extraction:** Extract 768-dimensional contextual token word embeddings from `bert-base-uncased` using attention-masked mean pooling over all valid non-padding tokens.
2. **Leak-Free Temporal Validation:** Enforce deterministic chronological sorting (`Creation_Date ASC, ID ASC`) with an 80%/20% development/test split and **5-fold expanding-window (walk-forward) temporal cross-validation** (195 total folds across 39 pairs).
3. **Multi-Family Regression Benchmarking:** Train and evaluate four distinct regression architectures—**Ridge Regression**, **Random Forest**, **Gradient Boosting (HistGradientBoosting)**, and a **3-Layer 1D CNN**—with strict training-fold feature scaling (`StandardScaler`).
4. **Skew-Robust Evaluation:** Report both **Mean Absolute Error (MAE)** and **Median Absolute Error (MdAE)** across all 5-fold validation runs and held-out cross-project test sets.

## Dataset Description

The experiments utilize the **TAWOS (The Agile Open Source)** dataset containing real-world Jira issue tracking records across diverse software engineering domains (cloud infrastructure, databases, IDEs, SDKs, and web platforms).

* **Total Valid Issues:** `63,003` (filtered for $\text{Story Points} > 0$, cleaned text length $\ge 5$ characters, zero missing text or target values).
* **Input Feature (`clean_text`):** Concatenated and preprocessed issue `Title + Description`.
* **Target Variable (`story_point`):** Continuous positive Agile story point estimates (Range: `0.25` to `26,710,114.0`).
* **Project Scope:** `39` software projects evaluated across `39` assigned cross-project pairs ($P_{\text{source}} \neq P_{\text{target}}$).

### Mandatory Canonical Project Mappings

To align TAWOS database schema names with the project guidelines, the following four mappings are strictly enforced:

| Guideline Project Name | TAWOS Database Name | Project ID | Project Key | Valid Issues ($\text{SP} > 0$) |
| :--- | :--- | :--- | :--- | :--- |
| **Evergreen** | The MongoDB Engineering | `43` | `EVG` | `5,069` |
| **C++ driver (Appcelerator)** | Aptana Studio | `8` | `APSTUD` | `538` |
| **C++ driver (MongoDB)** | The Mongo C++ driver | `31` | `CXX` | `212` |
| **Java driver** | The Mongo Java driver | `32` | `JAVA` | `238` |

## Repository Structure

```
├── code.ipynb                                      # Main end-to-end executable Jupyter Notebook
├── README.md                                       # Project documentation & reproduction guide
├── config/                                         # Configuration files (hyperparameters, seeds, splits)
├── src/                                            # Modular preprocessing, embedding & training scripts
├── data/
│   ├── processed/
│   │   ├── issues.csv                              # 63,003 cleaned & filtered TAWOS issues
│   │   ├── projects.csv                            # Registry of 39 software projects
│   │   └── project_mappings.csv                    # Canonical project mapping & 39 pair definitions
│   └── splits/
│       └── splits_manifest.json                    # Temporal split verification & fold indices
└── results/
    ├── embeddings/
    │   └── bert_embeddings.npy                     # Cached (63003, 768) float32 BERT embeddings
    ├── tables/
    │   ├── code_all_39_pairs_results.csv           # Granular metrics across all 39 pairs & 4 models
    │   └── comparative_results_with_cnn.csv        # Summary evaluation table (Section 3.4)
    └── plots/
        ├── eda_overview.png                        # 4-panel Exploratory Data Analysis visualization
        └── code_comparative_plots.png              # Comparative model evaluation plots
```

## Experimental Configuration & Hyperparameters

### 1. Environment & Reproducibility

* **Python Version:** `3.13.7`
* **Core Libraries:** `numpy==2.4.0`, `pandas==2.3.3`, `scikit-learn==1.8.0`, `matplotlib==3.10.8`, `seaborn`, `scipy`, `torch`, `transformers`
* **Global Random Seed:** `SEED = 42` (`np.random.seed(42)` and deterministic model `random_state=42`)

### 2. Temporal Split & Validation Settings

* **Chronological Sort Key:** `Creation_Date ASC, ID ASC`
* **Development / Held-Out Test Split:** `80%` Source Development / `20%` Target Held-Out Test ($\max(\text{Dev}) \le \min(\text{Test})$, zero ID overlap)
* **Cross-Validation Strategy:** `5-Fold Expanding-Window (Walk-Forward) CV` on the 80% development set (partitioned into 6 chronological windows $W = \lfloor N_{\text{dev}} / 6 \rfloor$; fold $k$ trains on windows $1 \dots k$ and validates on window $k+1$)
* **Feature Normalization:** `StandardScaler` fit strictly on each training fold before transforming validation/test splits

### 3. BERT Embedding Settings

* **Assigned Representation (`BERT`):** `bert-base-uncased` $\to$ Attention-masked token mean pooling over all valid non-padding tokens:
  $$\mathbf{e}_{\text{BERT}} = \frac{\sum_{l=1}^{L} m_l \mathbf{h}_l}{\sum_{l=1}^{L} m_l} \in \mathbb{R}^{768}$$
* **Vector Dimensionality:** `768` dimensions (`float32`)

### 4. Downstream Regression Models & Hyperparameters

| Model Family | Implementation / Architecture | Key Hyperparameters |
| :--- | :--- | :--- |
| **Ridge Regression** | L2-Regularized Linear Model | `alpha = 1.0` |
| **Random Forest** | Bagged Decision Tree Ensemble | `n_estimators = 40`, `max_depth = 12`, `random_state = 42` |
| **Gradient Boosting** | `HistGradientBoostingRegressor` / XGBoost | `max_iter = 40`, `max_depth = 4`, `learning_rate = 0.05`, Huber loss |
| **3-Layer CNN** | Sequential 1D CNN / Deep Regressor | `Conv1D(128, k=3)` + ReLU $\to$ `Conv1D(64, k=3)` + ReLU $\to$ `Dense(32)` + ReLU $\to$ `Dense(1, Linear)`, `dropout = 0.2` |

## Setup Instructions

### Option A: Local Workstation Setup

1. **Clone / Navigate to the Project Directory:**
   ```bash
   cd "NLP project"
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn scipy torch transformers jupyter
   ```

4. **Verify Data Files:**
   Ensure `issues.csv`, `projects.csv`, and `project_mappings.csv` are placed inside `data/processed/` (or in the root directory—the notebook automatically discovers and symlinks `issues.csv`).

### Option B: Kaggle Notebook Setup

1. Upload `code.ipynb` to Kaggle and attach the TAWOS dataset as an input data source (`/kaggle/input/...`).
2. Enable GPU acceleration (`T4` or `P100`) if recomputing raw BERT embeddings from scratch; otherwise, CPU mode executes rapidly using the cached `bert_embeddings.npy` tensor.
3. Run all cells sequentially—the setup cell automatically configures `/kaggle/working` directories and links `issues.csv`.

## Instructions for Reproducing Experiments & Demonstration

To reproduce all principal results, tables, and visualizations:

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook code.ipynb
   ```

2. **Execute Cells Sequentially (`Run All`):**
   * **Cell 1 (Environment Setup):** Verifies hardware, sets `SEED = 42`, creates `data/` and `results/` folders, and links `data/processed/issues.csv`.
   * **Cells 2–4 (Data Ingestion & Audit):** Loads the `63,003` valid issues across `39` projects and verifies compliance with the 4 mandatory project renaming rules.
   * **Cells 5–6 (Exploratory Data Analysis):** Generates the 4-panel EDA figure (`eda_overview.png`) and computes per-project story point summary statistics.
   * **Cell 7 (Temporal Split Verification):** Validates `data/splits/splits_manifest.json` across all `39` pairs and `195` walk-forward folds (`Zero Date Leakage = True`, `Zero ID Overlap = True`).
   * **Cell 8 (BERT Embedding Verification):** Loads and validates tensor shape and integrity for BERT `(63003, 768)` embeddings.
   * **Cells 9–10 (Model Training & 39-Pair Evaluation):** Trains Ridge, Random Forest, Gradient Boosting, and the 3-Layer CNN across all 39 pairs on BERT embeddings (`156` total model-pair evaluations) and exports `results/tables/code_all_39_pairs_results.csv`.
   * **Cells 11–14 (Evaluation Tables & Plots):** Prints the Section 3.4 BERT evaluation table, 5-fold walk-forward dynamics, model comparison plots (`results/plots/code_comparative_plots.png`), and top/bottom cross-project pair breakdowns.
   * **Cell 15 (Submission Integrity Audit):** Confirms all required dataset, embedding, table, and plot artifacts exist and are non-empty.

## Experimental Results

### 1. BERT Word Embeddings Evaluation Table (Section 3.4)

Aggregated across all **39 cross-project pairs** using **5-Fold Expanding-Window CV** ($\text{Mean} \pm \text{Std}$) and **Held-Out Test Evaluation**:

| Representation | Model Configuration | 5-Fold CV MAE ($\text{mean} \pm \text{std}$) | 5-Fold CV MdAE ($\text{mean} \pm \text{std}$) | Held-Out Test MAE | Held-Out Test MdAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT word embeddings** | Ridge Regression (`alpha=1.0`), 768-dim | $5.731 \pm 3.132$ | $3.475 \pm 1.737$ | $9.236$ | $5.346$ |
| **BERT word embeddings** | Random Forest (`n=40, depth=12`), 768-dim | $3.953 \pm 1.879$ | $2.952 \pm 1.153$ | $4.700$ | $3.509$ |
| **BERT word embeddings** | Gradient Boosting (`max_iter=40, depth=4`), 768-dim | $3.862 \pm 1.874$ | $2.793 \pm 1.158$ | $4.172$ | $3.167$ |
| **BERT word embeddings** | 3-Layer CNN (`Conv1D-128, Conv1D-64, FC-32`), 768-dim | $4.239 \pm 2.988$ | $2.731 \pm 2.013$ | $5.052$ | $2.749$ |

### 2. Five-Fold Expanding-Window Validation Breakdown (BERT + Gradient Boosting)

Mean Absolute Error (MAE) progression across the 5 chronological walk-forward folds:

| Representation | Model | Fold 1 MAE | Fold 2 MAE | Fold 3 MAE | Fold 4 MAE | Fold 5 MAE | 5-Fold Mean MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT (768-d)** | Gradient Boosting | $3.892$ | $3.841$ | $3.815$ | $3.784$ | $3.978$ | $3.862$ |

### 3. Cross-Project Pair Extremes (BERT + Gradient Boosting)

#### Top 5 Best-Performing Cross-Project Pairs (Lowest Held-Out Test MAE)

| Pair ID | Source $\to$ Target Project | Dev Size | Test Size | 5-Fold CV MAE | Held-Out Test MAE | Held-Out Test MdAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `33` | C++ driver (MongoDB) $\to$ Java driver | `100` | `9` | $0.560$ | $0.658$ | $0.592$ |
| `23` | Evergreen $\to$ Fabric | `2,836` | `126` | $0.787$ | $0.753$ | $0.629$ |
| `21` | FishEye $\to$ DotNetNuke Platform | `173` | `517` | $1.182$ | $0.851$ | $0.598$ |
| `20` | Indy Node $\to$ Software Cloud | `100` | `2` | $2.078$ | $1.140$ | $1.140$ |
| `26` | Crowd $\to$ Sawtooth | `170` | `190` | $1.019$ | $1.141$ | $0.792$ |

#### Top 5 Most Challenging Cross-Project Pairs (Highest Held-Out Test MAE)

| Pair ID | Source $\to$ Target Project | Dev Size | Test Size | 5-Fold CV MAE | Held-Out Test MAE | Held-Out Test MdAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `11` | Moodle $\to$ Atlassian Software Server | `179` | `66` | $22.658$ | $27.527$ | $16.486$ |
| `38` | Appcelerator Daemon $\to$ DataCass | `14` | `2` | $27.947$ | $19.714$ | $19.714$ |
| `25` | DataCass $\to$ Indy SDK | `172` | `142` | $9.905$ | $8.569$ | $6.331$ |
| `32` | C++ driver (Appcelerator) $\to$ Evergreen | `671` | `1,014` | $5.052$ | $7.224$ | $6.742$ |
| `8` | MXNet $\to$ The Titanium SDK | `164` | `611` | $4.697$ | $6.791$ | $6.184$ |

## Key Analytical Takeaways

* **Non-Linear & Deep Models Outperform Linear Baselines:** On BERT word embeddings, Gradient Boosting achieves the lowest overall held-out Test MAE (`4.172`) and 5-Fold CV MAE (`3.862 ± 1.874`), while the 3-Layer CNN achieves the lowest held-out Test MdAE (`2.749`) and 5-Fold CV MdAE (`2.731 ± 2.013`). Ridge Regression suffers from high extrapolation variance across disjoint project scales (`9.236` Test MAE).
* **Impact of Story Point Skewness:** Across all models trained on BERT representations, MdAE is consistently lower than MAE because extreme epic outliers inflate arithmetic mean errors, whereas MdAE accurately reflects typical sprint-level estimation error ($\approx 2.75\text{–}3.51$ story points).
