# Fraud Detection Model Card — phase2-1

Generated: 2026-10-17T10:29:26+00:00
Training framework: pure-Python GBDT (`fraud_detection/gbdt.py`), artifact `models/fraud_model.json`

## Intended use
Real-time risk scoring of payment transactions inside the transaction path.
Outputs a 0-100 risk index, an action (allow / log / step-up / hold / block),
and the top contributing signals for each decision.

## Data
- Synthetic transaction + behavioral stream from `fraud_detection/simulator.py`
  (the plan's documented fallback when no real transaction history exists).
- 15000 measured events after 2000 warm-up events,
  200 users, fraud rate 0.03.
- Chronological split: train 10500 / validation 2250 /
  test 2250 (no random shuffling — avoids look-ahead leakage).
- Features are computed by the same `FeatureEngine` used in production scoring,
  so offline/online parity holds by construction.

## Model
- **Model 1** — gradient-boosted trees on transaction context:
  60 trees, depth 3,
  32 features, class-weighted for fraud prevalence.
- **Model 2** — online behavioral-biometrics anomaly detector (per-user baseline
  over keystroke/mouse aggregates). Unsupervised; no labels used.
- **Model 3** — logistic classifier over sender→recipient graph features
  (distinct payers, hourly growth, shared devices/IPs); ring flag adds +15 risk.
- Calibration: Platt scaling fitted on the validation split.

## Held-out performance
| Metric | Value |
|---|---|
| ROC-AUC | 0.997 |
| PR-AUC | 0.9424 |
| Precision @ 80% recall | 1.0 |
| Precision @ 90% recall | 0.7195 |
| Calibration ECE | 0.0173 |
| Pipeline capture rate (risk >= 60) | 0.8308 |
| Pipeline false positive rate | 0.0032 |
| Ring model threshold / AUC | 0.8 / 1.0 |
| Scoring latency p50 / p99 | 0.105 / 0.3 ms |

## Limitations
- Trained on synthetic data; absolute performance will not transfer to real
  portfolios without retraining on historical labeled transactions.
- The graph model uses hand-built structural features, not Node2Vec/GraphSAGE
  embeddings — an intentional dependency-free simplification.
- Behavioral baselines need several sessions per user; new accounts fall back
  to transaction-only scoring (`behavior_missing_flag`).
- Model 1 is a reference GBDT implementation, not LightGBM/XGBoost; retrain with
  LightGBM offline when the training environment allows it.
- No ONNX export yet; the JSON artifact is the deployment format for the demo.
