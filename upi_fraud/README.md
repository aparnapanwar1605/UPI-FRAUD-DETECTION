# UPI Fraud Detection under Extreme Class Imbalance

Detecting fraudulent UPI transactions when **only 1 in 1,000 is fraud**, and comparing the main ways of handling that imbalance: plain models, **cost-sensitive learning**, **SMOTE / undersampling**, and **unsupervised anomaly detection**.

> **Data note:** real UPI fraud data is private, so transactions are **simulated** (`src/data_generator.py`). The simulator includes three fraud playbooks, *stealth* fraud that imitates normal behaviour, and *hard-negative* legitimate transactions that look suspicious. Absolute metrics are illustrative; the methodology is what transfers to real data.

## Highlights

- **Time-based split** (train → validation → test) instead of a random split, to mimic real deployment.
- **Threshold chosen on validation only**; the test set is used once at the end.
- **Resampling inside the CV/pipeline** (`imblearn`), so no synthetic samples leak into validation folds.
- **Business-aware evaluation:** PR-AUC, precision/recall at an alert budget, and **net benefit in ₹** (fraud stopped − review cost), with sensitivity to review cost.
- **Bootstrap confidence interval** on test PR-AUC, since the test set has few frauds.
- **Failure analysis:** which fraud types are missed and who gets falsely flagged.

## Results (seed 42)

| Approach | Validation PR-AUC | 5-fold CV PR-AUC |
|---|---|---|
| LightGBM (plain) | **0.403** | **0.362** |
| LightGBM + `scale_pos_weight` | 0.361 | 0.318 |
| Random Forest + class weights | 0.350 | – |
| Logistic (plain) | 0.287 | 0.205 |
| LightGBM + SMOTE | 0.271 | 0.273 |
| Logistic + SMOTE | 0.221 | 0.187 |
| Logistic + class weights | 0.201 | 0.167 |
| Logistic + undersampling | 0.192 | 0.177 |
| Isolation Forest (unsupervised) | 0.072 | – |
| Random guessing | 0.001 | – |

**Final test set (untouched):** PR-AUC **0.31** (95% CI 0.21 – 0.42) ≈ 360× random. At a ₹25 review cost the model flags ~0.95% of transactions, catches 67% of frauds (**88% of fraud value**), for a net benefit of ≈ ₹2.8 lakh over 20 days.

**Key findings**
1. Accuracy is meaningless here (a "never fraud" model scores 99.9%).
2. **Resampling did not help**: plain LightGBM beat its weighted and SMOTE variants. With ~250 training frauds, SMOTE creates unrealistic synthetic points; a cost-based threshold does the job more cleanly.
3. **Model family mattered more than the imbalance trick** (trees beat linear models because fraud is a *combination* of weak signals).
4. Unsupervised anomaly detection is a weak substitute for labels, but far above random.
5. Remaining errors are structural: **stealth fraud** (~17% recall vs ~90% for obvious fraud) and genuinely unusual customers (>50% of false alarms) need new data signals, not new algorithms.

## Project structure

```
upi_fraud/
├── upi_fraud_detection.ipynb   # main notebook (outputs included)
├── src/
│   ├── data_generator.py       # synthetic UPI transaction simulator
│   └── features.py             # feature engineering
├── models/                     # saved model + threshold (created by the notebook)
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook upi_fraud_detection.ipynb
```
Everything is generated locally (no downloads) and is reproducible via a fixed seed. Full run takes ~2–3 minutes.

## Limitations

- Simulated data; real fraud is adversarial and non-stationary.
- Only ~76 frauds in the test window, so confidence intervals are wide.
- No label delay, graph features or drift monitoring.
- Scores from weighted/resampled models are not calibrated probabilities.
- Review cost and "fraud stopped = full amount" are simplifying assumptions.

## Next steps

Time-series CV and retraining, graph features for mule rings, probability calibration, SHAP explanations per alert, a FastAPI/Streamlit scoring service, and validating the pipeline on a real public fraud dataset.
