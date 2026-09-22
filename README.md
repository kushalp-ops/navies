# navies# MAGIC Gamma Telescope Classification with Naive Bayes

A Machine Learning project using **Gaussian Naive Bayes** to classify high-energy gamma particles versus hadron background noise based on Cherenkov gamma telescope data.

---

## 📌 Project Overview
This repository contains a Python implementation and data pipeline for classifying atmospheric Cherenkov telescope events from the **MAGIC Gamma Telescope** dataset. The pipeline handles data preprocessing, feature scaling, class rebalancing, model training, and evaluation.

- **Primary Model**: Gaussian Naive Bayes (`GaussianNB`)
- **Core Concept**: Bayes Theorem & Conditional Probability
- **Target Classes**: 
  - `g` (Gamma signal) $\rightarrow$ Mapped to `1`
  - `h` (Hadron background) $\rightarrow$ Mapped to `0`

---

## 📊 Dataset Information
- **Source**: MAGIC Gamma Telescope Dataset (`magic04.data`)[cite: 4]
- **Total Samples**: 19,020 instances[cite: 4]
- **Total Features**: 10 continuous numeric attributes + 1 class target[cite: 4]

### Feature List:
1. `fLength`: Major axis of ellipse [mm]
2. `fWidth`: Minor axis of ellipse [mm]
3. `fSize`: 10-log of sum of content of all pixels [in #phot]
4. `fConc`: Ratio of sum of two highest pixels over size
5. `fConc1`: Ratio of highest pixel over size
6. `fAsym`: Distance from highest pixel to center, projected onto major axis [mm]
7. `fM3Long`: 3rd root of 3rd moment along major axis [mm]
8. `fM3Trans`: 3rd root of 3rd moment along minor axis [mm]
9. `fAlpha`: Angle of major axis with vector to camera center [deg]
10. `fDist`: Distance from origin to center of ellipse [mm]
11. `class`: Target label (`g` or `h`)

---

## 🛠️ Pipeline Architecture

1. **Data Loading & Preprocessing**:
   - Read dataset without header and assign standard feature column names.
   - Encode binary target values (`g` = `1`, `h` = `0`).

2. **Data Splitting**:
   - Randomly partitioned into Train, Validation, and Test sets (60% / 20% / 20%).

3. **Feature Scaling & Class Balancing**:
   - **StandardScaler**: Standardizes features to zero mean and unit variance.
   - **RandomOverSampler**: Over-samples minority class instances in the training set to resolve class imbalance.

4. **Model Training & Evaluation**:
   - Train `GaussianNB()` on balanced training set.
   - Evaluate model metrics using classification reports.

---

## 💻 Tech Stack & Dependencies

- **Language**: Python 3.x
- **Libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imbalanced-learn` (`imblearn`)
  - `matplotlib`

---

## ⚡ Quick Start

```bash
# Clone repository
git clone [https://github.com/your-username/magic-gamma-naive-bayes.git](https://github.com/your-username/magic-gamma-naive-bayes.git)
cd magic-gamma-naive-bayes

# Install required packages
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
