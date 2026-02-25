# Baseline Model Comparison

## Problem Statement
Predict purchase intent from the first 5 clicks of a user session in an e-commerce site.

## Dataset
- **Total Sessions**: 24,026
- **Total Clicks**: 165,474  
- **Class Distribution**: 
  - No Purchase Intent (0): 20,027 (83.4%)
  - Purchase Intent (1): 3,999 (16.6%)
- **Train/Test Split**: 80/20 (stratified)
- **Features**: 27 session-level features

## Label Strategy
**ProxyHybridIntentLabelStrategy**: Sessions with ≥8 clicks AND ≥50% high-price items are labeled as intent=1

## Models Trained

### 1. Logistic Regression (Baseline)
- **Model Type**: Linear classifier with balanced class weights
- **Iterations**: 1,000 (max_iter)

**Performance Metrics:**
| Metric | Score |
|--------|-------|
| **Accuracy** | 0.7528 |
| **Precision** | 0.3985 |
| **Recall** | 0.9525 |
| **F1-Score** | 0.5619 |
| **ROC-AUC** | 0.8913 |

**Confusion Matrix:**
```
              Predicted
              No(0)  Yes(1)
Actual No(0)  2856   1150
       Yes(1)   38    762
```

**Analysis:**
- High recall (95.25%) - catches most purchase intent sessions
- Low precision (39.85%) - many false positives
- Good at identifying potential buyers but over-predicts

---

### 2. Random Forest (Baseline)
- **Model Type**: Ensemble of 100 trees, max_depth=10, balanced class weights
- **Parallel Jobs**: All cores (-1)

**Performance Metrics:**
| Metric | Score |
|--------|-------|
| **Accuracy** | 0.7834 |
| **Precision** | 0.4286 |
| **Recall** | 0.9038 |
| **F1-Score** | 0.5814 |
| **ROC-AUC** | 0.8920 |

**Confusion Matrix:**
```
              Predicted
              No(0)  Yes(1)
Actual No(0)  3042    964
       Yes(1)   77    723
```

**Analysis:**
- Better balanced performance than Logistic Regression
- Slightly better precision (42.86%) with good recall (90.38%)
- **Best baseline model** - higher accuracy and F1-score

---

## Comparison Summary

| Metric | Logistic Regression | Random Forest | Winner |
|--------|-------------------|---------------|--------|
| Accuracy | 0.7528 | **0.7834** | RF |
| Precision | 0.3985 | **0.4286** | RF |
| Recall | **0.9525** | 0.9038 | LR |
| F1-Score | 0.5619 | **0.5814** | RF |
| ROC-AUC | 0.8913 | **0.8920** | RF |

## Key Insights

1. **Class Imbalance**: Dataset is heavily imbalanced (83% vs 17%)
2. **High Recall Priority**: Both models prioritize recall over precision (using balanced class weights)
3. **Random Forest Performs Better**: RF achieves better balance between precision and recall
4. **ROC-AUC**: Both models have similar strong discrimination ability (~0.89)
5. **False Positives**: Major challenge - about 60% of predicted purchases are false positives

## Recommendations

**For Production Use:**
- **Random Forest** is the recommended baseline model
- Achieves 78.34% accuracy with 90.38% recall
- Better suited for scenarios where catching potential buyers is critical

**For Future Improvements:**
1. Feature engineering: Add temporal patterns, sequence features
2. Hyperparameter tuning: Grid/random search for optimal parameters
3. Advanced models: Gradient boosting (XGBoost, LightGBM), neural networks
4. Threshold optimization: Adjust prediction threshold based on business costs
5. SMOTE/undersampling: Address class imbalance
6. Feature selection: Identify most important features

## Files Generated
- `baseline_model.pkl` - Logistic Regression model
- `baseline_rf_model.pkl` - Random Forest model  
- `baseline_model_metrics.txt` - LR metrics
- `baseline_rf_model_metrics.txt` - RF metrics
- `features_first_n.csv` - Processed session features for training
- `baseline_labels.csv` - Generated labels
