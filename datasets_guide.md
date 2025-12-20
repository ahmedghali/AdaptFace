# Dataset Download Guide - AdaptFace Project

This guide provides download links and instructions for all required datasets.

---

## Required Datasets Overview

| Dataset          | Purpose         | Size   | Identities | Images  |
|------------------|-----------------|--------|------------|---------|
| CASIA-WebFace    | Training        | ~4GB   | 10,575     | 494,414 |
| LFW              | Validation      | ~173MB | 5,749      | 13,233  |
| CFP-FP           | Cross-pose eval | ~200MB | 500        | 7,000   |
| AgeDB-30         | Cross-age eval  | ~450MB | 568        | 16,488  |
| CALFW            | Cross-age eval  | ~200MB | 4,025      | 12,174  |
| CPLFW            | Cross-pose eval | ~200MB | 3,884      | 11,652  |

### Optional/Advanced Datasets
| Dataset | Purpose | Notes |
|---------|---------|-------|
| IJB-B/IJB-C | Large-scale eval | Requires NIST agreement |
| Thermal (NIST/UND) | Cross-domain | Academic access required |
| SCface | Surveillance | Academic access required |

---

## 1. CASIA-WebFace (Training Dataset)

**Status:** [ ] Downloaded

**Description:** Main training dataset with 10K+ identities. Clean, aligned faces.

**Download Options:**

### Option A: From InsightFace (Recommended)
```bash
# Download cleaned version from InsightFace
# Link: https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_
# File: CASIA-WebFace (or faces_webface_112x112.zip for aligned version)
```

### Option B: Academic Request
- Original site: http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html
- Requires academic institution email

### Option C: Kaggle Mirror
```bash
# https://www.kaggle.com/datasets/debarghamitraroy/casia-webface
# May require Kaggle account
```

**After Download:**
```bash
# Extract to data/casia-webface/
# Expected structure:
# data/casia-webface/
#   ├── 0000001/
#   │   ├── 001.jpg
#   │   ├── 002.jpg
#   │   └── ...
#   ├── 0000002/
#   └── ...
```

---

## 2. LFW (Labeled Faces in the Wild)

**Status:** [ ] Downloaded

**Description:** Standard face verification benchmark. 6,000 face pairs for testing.

**Download:**
```bash
# Official site: http://vis-www.cs.umass.edu/lfw/

# Direct download (aligned with deep funneling):
# http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz

# Or via wget:
wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
tar -xzf lfw-deepfunneled.tgz -C data/
mv data/lfw-deepfunneled data/lfw
```

**Pairs file (for evaluation):**
```bash
wget http://vis-www.cs.umass.edu/lfw/pairs.txt -O data/lfw/pairs.txt
```

---

## 3. CFP-FP (Celebrities in Frontal-Profile)

**Status:** [ ] Downloaded

**Description:** Cross-pose evaluation with frontal-profile pairs.

**Download:**
```bash
# Official site: http://www.cfpw.io/
# Requires registration

# After registration, download and extract to:
# data/cfp-fp/
```

**Alternative:** Available in InsightFace evaluation package.

---

## 4. AgeDB-30

**Status:** [ ] Downloaded

**Description:** Cross-age evaluation with 30-year age gaps.

**Download:**
```bash
# Part of InsightFace evaluation package
# https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_

# Or direct:
# https://drive.google.com/file/d/1AoZrZfym5ZhdTyKSxD0qxa7Xrp2Q1ftp/view
```

---

## 5. CALFW (Cross-Age LFW)

**Status:** [ ] Downloaded

**Description:** Cross-age variant of LFW.

**Download:**
```bash
# Official: http://whdeng.cn/CALFW/
# Requires academic email registration

# Extract to: data/calfw/
```

---

## 6. CPLFW (Cross-Pose LFW)

**Status:** [ ] Downloaded

**Description:** Cross-pose variant of LFW.

**Download:**
```bash
# Official: http://whdeng.cn/CPLFW/
# Requires academic email registration

# Extract to: data/cplfw/
```

---

## Quick Setup: InsightFace Evaluation Package

InsightFace provides a convenient package with multiple evaluation datasets pre-aligned:

```bash
# Download from:
# https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_

# This includes:
# - LFW
# - CFP-FP
# - AgeDB-30
# - CALFW
# - CPLFW

# In .bin format - we'll need to convert to images
```

---

## Expected Folder Structure

After downloading all datasets, your folder structure should look like:

```
d:\PAPERS\project_01\
├── data/
│   ├── casia-webface/           # Training data
│   │   ├── 0000001/
│   │   ├── 0000002/
│   │   └── ...
│   ├── lfw/                     # Validation
│   │   ├── pairs.txt
│   │   ├── Aaron_Eckhart/
│   │   └── ...
│   ├── cfp-fp/                  # Cross-pose eval
│   ├── agedb-30/                # Cross-age eval
│   ├── calfw/                   # Cross-age eval
│   ├── cplfw/                   # Cross-pose eval
│   └── processed/               # Preprocessed data (created by pipeline)
│       ├── train/
│       ├── val/
│       └── domain_splits/
├── src/
├── configs/
└── ...
```

---

## Notes

### Dataset Licensing
- Most face datasets require academic use agreement
- Do NOT redistribute datasets
- Cite original papers when publishing

### Storage Requirements
- Raw datasets: ~10-15GB
- Processed datasets: ~20-30GB
- Total recommended: 50GB free space

### Next Steps After Download
1. Run preprocessing pipeline (creates aligned 224x224 images)
2. Create domain splits from CASIA
3. Verify data loading works

---

*Created: 2025-12-10*
