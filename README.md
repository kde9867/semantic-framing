# Semantic Framing as a Demographic Signal

Micro–Macro Shifts in Fertility Discourse Across the COVID-19 Pandemic

This repository contains the implementation of BERT-based models for analyzing semantic shifts in Korean fertility discourse across different time periods (pre-COVID, during-COVID, post-COVID).

## Repository Structure

```         
semantic-framing/
├── data/                  # Training/validation/test datasets
│   ├── Full dataset (all periods combined)
│   ├── Period-split datasets (range1, range2, range3)
│   └── Quarterly-split datasets (by quarter)
├── model/                 # Training scripts
│   ├── train_full_model.py           # Train single model on all data
│   ├── train_period_models.py        # Train separate models per period
│   └── train_quarterly_models.py     # Train separate models per quarter
├── semantic-axis/         # Semantic projection analysis
│   ├── Micro-macro axis construction
│   ├── Period-based projection analysis
│   └── Quarterly projection analysis
└── README.md
```

## Dataset Structure

-   **Full dataset**: Combined data across all time periods
-   **Period datasets**:
    -   `range1`: Pre-COVID period
    -   `range2`: During-COVID period
    -   `range3`: Post-COVID period
-   **Quarterly datasets**: Data split by quarters (YYYYQX format)

## Training Scripts

### 1. Full-model Training

Train a single model on the entire dataset:

``` bash
python model/train_full_model.py \
    --data_path1 data/train.csv \
    --data_path2 data/val.csv \
    --data_path3 data/test.csv \
    --epochs 3 \
    --batch_size 16 \
    --gpu_id 0
```

### 2. Period-based Training

Train separate models for each COVID-19 period (pre/during/post):

``` bash
python model/train_period_models.py \
    --data_path1 data/range1.csv \
    --data_path2 data/range2.csv \
    --data_path3 data/range3.csv \
    --test_data_path data/test.csv \
    --epochs 3 \
    --batch_size 16 \
    --gpu_id 0
```

### 3. Quarterly Training

Train separate models for each quarter:

``` bash
python model/train_quarterly_models.py \
    --data_path1 data/range1.csv \
    --data_path2 data/range2.csv \
    --data_path3 data/range3.csv \
    --date_column date \
    --test_data_path data/test.csv \
    --epochs 3 \
    --batch_size 16 \
    --gpu_id 0
```

## Semantic Axis Analysis

The `semantic-axis/` directory contains scripts for:

1.  **Axis Construction**: Build micro-macro semantic axis from labeled texts
2.  **Period Projection**: Project target words onto the axis using period-specific models
3.  **Quarterly Projection**: Fine-grained temporal analysis using quarterly models

### Key Components

-   **Micro texts**: Individual-level, personal experiences related to fertility
-   **Macro texts**: Societal-level, policy-oriented discussions
-   **Projection analysis**: Track how word meanings shift along the micro-macro axis over time

### Target Keywords

109 fertility-related keywords including: - Core terms: birth, childcare, pregnancy - Policy terms: policy, support, welfare 
## Requirements

```         
transformers>=4.30.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
```
