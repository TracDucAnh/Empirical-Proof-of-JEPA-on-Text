# Run Commands

## 1. Go To Project

```bash
cd /home/khang.nhat/codes/Empirical-Proof-of-JEPA-on-Text
```

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

## 3. Download Pretraining C4 Data

### BERT MLM+NSP Document Cache

```bash
python pretrained_data_sampler.py --mode bert --max-length 256
```

### Single-Sentence Cache For JEPA/BYOL/VICReg/Barlow Twins

```bash
python pretrained_data_sampler.py --mode sentences
```

### All Pretraining Caches

```bash
python pretrained_data_sampler.py --mode all
```

## 4. Pretrain Methods

### BERT

```bash
python pretrain/pretrained_BERT.py
```

### JEPA

```bash
python pretrain/pretrained_JEPA.py
```

### BYOL

```bash
python pretrain/pretrained_BYOL.py
```

### VICReg

```bash
python pretrain/pretrained_VICReg.py
```

### Barlow Twins

```bash
python pretrain/pretrained_Barlow_Twins.py
```

## 5. Download Downstream Data

### Classification

```bash
python downstream_data_downloader.py --task classification
```

### QA

```bash
python downstream_data_downloader.py --task qa
```

### Retrieval

```bash
python downstream_data_downloader.py --task retrieval
```

### All Downstream Tasks

```bash
python downstream_data_downloader.py --task all
```

## 6. Evaluate BERT

### Classification

```bash
python downstream_evaluator.py --task classification --method bert --checkpoint outputs/bert_pretraining/bert_pretraining_best.pt --epochs 5
```

### QA

```bash
python downstream_evaluator.py --task qa --method bert --checkpoint outputs/bert_pretraining/bert_pretraining_best.pt --epochs 5
```

### Retrieval

```bash
python downstream_evaluator.py --task retrieval --method bert --checkpoint outputs/bert_pretraining/bert_pretraining_best.pt --epochs 5 --retrieval-eval-k 10
```

### SNLI

```bash
python downstream_evaluator.py \
    --task nli \
    --method bert \
    --checkpoint outputs/bert_pretraining/bert_pretraining_best.pt \
    --epochs 5
```

## 7. Evaluate JEPA

### Classification

```bash
python downstream_evaluator.py --task classification --method jepa --checkpoint outputs/text_jepa/text_jepa_best.pt --epochs 5
```

### QA

```bash
python downstream_evaluator.py --task qa --method jepa --checkpoint outputs/text_jepa/text_jepa_best.pt --epochs 5
```

### Retrieval

```bash
python downstream_evaluator.py --task retrieval --method jepa --checkpoint outputs/text_jepa/text_jepa_best.pt --epochs 5 --retrieval-eval-k 10
```

### SNLI

```bash
python downstream_evaluator.py \
    --task nli \
    --method jepa \
    --checkpoint outputs/text_jepa/text_jepa_best.pt \
    --epochs 5
```

## 8. Evaluate BYOL

### Classification

```bash
python downstream_evaluator.py --task classification --method byol --checkpoint outputs/text_byol/text_byol_best.pt --epochs 5
```

### QA

```bash
python downstream_evaluator.py --task qa --method byol --checkpoint outputs/text_byol/text_byol_best.pt --epochs 5
```

### Retrieval

```bash
python downstream_evaluator.py --task retrieval --method byol --checkpoint outputs/text_byol/text_byol_best.pt --epochs 5 --retrieval-eval-k 10
```

### SNLI

```bash
python downstream_evaluator.py \
    --task nli \
    --method byol \
    --checkpoint outputs/text_byol/text_byol_best.pt \
    --epochs 5
```

## 9. Evaluate VICReg

### Classification

```bash
python downstream_evaluator.py --task classification --method vicreg --checkpoint outputs/text_vicreg/text_vicreg_best.pt --epochs 5
```

### QA

```bash
python downstream_evaluator.py --task qa --method vicreg --checkpoint outputs/text_vicreg/text_vicreg_best.pt --epochs 5
```

### Retrieval

```bash
python downstream_evaluator.py --task retrieval --method vicreg --checkpoint outputs/text_vicreg/text_vicreg_best.pt --epochs 5 --retrieval-eval-k 10
```

### SNLI

```bash
python downstream_evaluator.py \
    --task nli \
    --method vicreg \
    --checkpoint outputs/text_vicreg/text_vicreg_best.pt \
    --epochs 5
```

## 10. Evaluate Barlow Twins

### Classification

```bash
python downstream_evaluator.py --task classification --method barlow_twins --checkpoint outputs/text_barlow_twins/text_barlow_twins_best.pt --epochs 5
```

### QA

```bash
python downstream_evaluator.py --task qa --method barlow_twins --checkpoint outputs/text_barlow_twins/text_barlow_twins_best.pt --epochs 5
```

### Retrieval

```bash
python downstream_evaluator.py --task retrieval --method barlow_twins --checkpoint outputs/text_barlow_twins/text_barlow_twins_best.pt --epochs 5 --retrieval-eval-k 10
```

### SNLI

```bash
python downstream_evaluator.py \
    --task nli \
    --method barlow_twins \
    --checkpoint outputs/text_barlow_twins/text_barlow_twins_best.pt \
    --epochs 5
```
