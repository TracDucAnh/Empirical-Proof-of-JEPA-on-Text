# Run Commands

## 1. Go To Project

```bash
cd /home/khang.nhat/codes/Empirical-Proof-of-JEPA-on-Text
```

## 2. Install Requirements

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/pip install -r requirements.txt
```

## 3. Pretrain Methods

### BERT

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python pretrain/pretrained_BERT.py
```

### JEPA

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python pretrain/pretrained_JEPA.py
```

### BYOL

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python pretrain/pretrained_BYOL.py
```

### VICReg

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python pretrain/pretrained_VICReg.py
```

### Barlow Twins

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python pretrain/pretrained_Barlow_Twins.py
```

## 4. Download Downstream Data

### Classification

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_data_downloader.py --task classification
```

### QA

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_data_downloader.py --task qa
```

### Retrieval

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_data_downloader.py --task retrieval
```

### All Downstream Tasks

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_data_downloader.py --task all
```

## 5. Evaluate BERT

### Classification

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task classification --method bert --checkpoint outputs/bert_pretraining/bert_pretraining_latest.pt --epochs 5
```

### QA

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task qa --method bert --checkpoint outputs/bert_pretraining/bert_pretraining_latest.pt --epochs 5
```

### Retrieval

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task retrieval --method bert --checkpoint outputs/bert_pretraining/bert_pretraining_latest.pt --epochs 5 --retrieval-eval-k 10
```

## 6. Evaluate JEPA

### Classification

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task classification --method jepa --checkpoint outputs/text_jepa/text_jepa_latest.pt --epochs 5
```

### QA

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task qa --method jepa --checkpoint outputs/text_jepa/text_jepa_latest.pt --epochs 5
```

### Retrieval

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task retrieval --method jepa --checkpoint outputs/text_jepa/text_jepa_latest.pt --epochs 5 --retrieval-eval-k 10
```

## 7. Evaluate BYOL

### Classification

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task classification --method byol --checkpoint outputs/text_byol/text_byol_latest.pt --epochs 5
```

### QA

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task qa --method byol --checkpoint outputs/text_byol/text_byol_latest.pt --epochs 5
```

### Retrieval

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task retrieval --method byol --checkpoint outputs/text_byol/text_byol_latest.pt --epochs 5 --retrieval-eval-k 10
```

## 8. Evaluate VICReg

### Classification

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task classification --method vicreg --checkpoint outputs/text_vicreg/text_vicreg_latest.pt --epochs 5
```

### QA

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task qa --method vicreg --checkpoint outputs/text_vicreg/text_vicreg_latest.pt --epochs 5
```

### Retrieval

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task retrieval --method vicreg --checkpoint outputs/text_vicreg/text_vicreg_latest.pt --epochs 5 --retrieval-eval-k 10
```

## 9. Evaluate Barlow Twins

### Classification

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task classification --method barlow_twins --checkpoint outputs/text_barlow_twins/text_barlow_twins_latest.pt --epochs 5
```

### QA

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task qa --method barlow_twins --checkpoint outputs/text_barlow_twins/text_barlow_twins_latest.pt --epochs 5
```

### Retrieval

```bash
/home/khang.nhat/anaconda3/envs/llm/bin/python downstream_evaluator.py --task retrieval --method barlow_twins --checkpoint outputs/text_barlow_twins/text_barlow_twins_latest.pt --epochs 5 --retrieval-eval-k 10
```
