# Stage 3 Final Evaluation Report

This report documents the final evaluation of all baseline and GNN models on the strictly held-out test set (time steps 40–49).

## Methodology
- **Test set:** Time steps 40–49 (untouched during training/tuning).
- **Primary metric:** PR-AUC on the illicit class (handling ~2% class imbalance).
- **GNN variants:** Evaluated on both raw features (166 anonymized) and combined features (+ 8 engineered).
- **Scaling:** Engineered features were scaled using a `StandardScaler` fit *only* on the train split (time steps 1–34) to prevent data leakage.

## Final Results (Test Set)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |
| GraphSAGE (raw features) | | | | | |
| GraphSAGE (+ engineered) | | | | | |
| GAT (raw features) | | | | | |
| GAT (+ engineered) | | | | | |

*Note: Table will be populated after Person A provides the engineered features and sweeps are completed.*

## Hyperparameter Sweeps
*To be filled after running sweeps:*
- **GraphSAGE best config:** ...
- **GAT best config:** ...

## Conclusions
*To be filled.*
