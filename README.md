# Semantic Framing as a Demographic Signal  
## Micro–Macro Shifts in Fertility Discourse Across the COVID-19 Pandemic

This repository contains the implementation of **BERT-based models** for analyzing semantic shifts in **Korean fertility discourse** across different time periods, as described in our methodology.

---

## Training Scripts

### Full-model Training
```bash
python train_period_models.py \
    --data_path1 /path/to/train.csv \
    --data_path2 /path/to/val.csv \
    --data_path3 /path/to/test.csv \
    --epochs 3 \
    --batch_size 16 \
    --gpu_id 0
```
### Period-based Training
```bash
python train_period_models.py \
    --data_path1 /path/to/range1.csv \
    --data_path2 /path/to/range2.csv \
    --data_path3 /path/to/range3.csv \
    --test_data_path /path/to/test.csv \
    --epochs 3 \
    --batch_size 16 \
    --gpu_id 0
```
### Period-based Training
```bash
python train_quarterly_models.py \
    --data_path1 /path/to/range1.csv \
    --data_path2 /path/to/range2.csv \
    --data_path3 /path/to/range3.csv \
    --date_column date \
    --test_data_path /path/to/test.csv \
    --epochs 3 \
    --batch_size 16 \
    --gpu_id 0
```
