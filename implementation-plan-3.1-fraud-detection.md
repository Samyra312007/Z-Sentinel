# Implementation Plan: Real-Time Payment Fraud Detection with Behavioral Biometrics

## 1. Project Overview

**Problem:** Fraudulent transactions cost billions annually. Traditional batch ML models analyze transactions after the fact—by then, funds are gone. Real-time scoring inside the transaction path prevents loss before it happens.

**Solution:** A real-time fraud detection system that scores every payment transaction in milliseconds using:
- Transaction context features (amount, location, device, time, recipient risk)
- Behavioral biometrics (typing rhythm, mouse dynamics, navigation patterns)
- Graph-based fraud ring detection
- Streaming ML models deployed on IBM Z via ML for z/OS

**IBM Z Value Proposition:**
- Transactional AI: ML inference runs inside the transaction path on IBM Z
- Data never leaves the secure mainframe environment
- Atomic fraud decisions with transaction integrity
- Handles millions of transactions per second
- Full audit trail for compliance

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        IBM Z (z/OS)                             │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │  Transaction    │───▶│  Feature        │───▶│  ML for     │ │
│  │  Stream (CICS/  │    │  Engineering     │    │  z/OS       │ │
│  │  IMS/DC)        │    │  (Real-time     │    │  Inference  │ │
│  │                 │    │   feature calc) │    │  (Scoring)  │ │
│  └─────────────────┘    └─────────────────┘    └──────┬──────┘ │
│                                                        │        │
│  ┌─────────────────┐    ┌─────────────────┐           │        │
│  │  Behavioral     │───▶│  Biometric      │───────────┘        │
│  │  Biometrics     │    │  Feature Store  │                    │
│  │  Collection     │    │  (Redis/IMDB)   │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                        ┌──────────────────────┘ │
│                                        │                      │
│  ┌─────────────────┐    ┌─────────────┴──────┐               │
│  │  Decision        │◀───│  Fraud Score +     │               │
│  │  Engine          │    │  Explanation       │               │
│  │  (Allow/Step-up/ │    │  (Top signals)     │               │
│  │  Block/Review)   │    └────────────────────┘               │
│  └─────────────────┘                                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  IBM Z Security Layer                                     │  │
│  │  - Encrypted data at rest & in transit                   │  │
│  │  - CPACF hardware crypto                                  │  │
│  │  - Secure audit logging (immutable)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────┐
                        │  Model Training     │
                        │  Pipeline (Offline) │
                        │                     │
                        │  - Historical txns  │
                        │  - Graph features   │
                        │  - Model retraining │
                        └─────────────────────┘
```

---

## 3. Phase-by-Phase Implementation

### Phase 1: Foundation & Data (Weeks 1-2)

#### 1.1 Environment Setup
- [ ] Provision IBM Z environment (IBM Z Xplore for learning, or z/OS LPAR for development)
- [ ] Set up z/OS Connect EE for API exposure (if needed)
- [ ] Configure ML for z/OS environment
- [ ] Set up development tools: IBM Z Open Editor, Zowe CLI

#### 1.2 Data Sources & Ingestion
| Data Source | Type | Latency | Use |
|-------------|------|---------|-----|
| Payment transaction stream | Structured | Real-time | Primary features (amount, sender, receiver, channel, time) |
| Device fingerprinting | Structured | Real-time | Device ID, IP, geo, browser/OS |
| Behavioral biometrics | Time-series | Real-time | Keystroke dynamics, mouse movement, touch patterns |
| Historical fraud labels | Labels | Batch | Model training (approved/declined/fraud_confirmed) |
| External risk feeds | Enrichment | Near-real-time | Sender/receiver blacklists, geo-risk scores |

#### 1.3 Data Pipeline Setup
- [ ] Use IBM Broker for z/OS or Kafka on z/OS for transaction streaming
- [ ] Build real-time feature computation:
  - [ ] Transaction amount deviation from user's 30-day average
  - [ ] Time since last transaction
  - [ ] Location distance from last transaction
  - [ ] Device change flag
  - [ ] New recipient flag
  - [ ] Velocity checks (txns per hour/day)
- [ ] Behavioral feature extraction:
  - [ ] Keystroke duration/interval statistics
  - [ ] Mouse movement entropy
  - [ ] Session behavior consistency score vs. user profile

#### 1.4 IBM Z-Specific Setup
- [ ] Configure transactional context propagation (correlation ID across systems)
- [ ] Set up secure credential management (z/OS RACF, keyrings)
- [ ] Configure audit logging for all fraud decisions

---

### Phase 2: ML Model Development (Weeks 3-5)

#### 2.1 Problem Framing
- [ ] **Task:** Binary classification (fraud/not fraud) + risk score (0-100)
- [ ] **Latency target:** < 50ms inference time (must fit within transaction timeout)
- [ ] **Evaluation metrics:**
  - [ ] Primary: Precision at fixed recall (catch X% of fraud with minimal false positives)
  - [ ] Secondary: ROC-AUC, PR-AUC, latency p99
  - [ ] Business: False positive rate (customer friction), fraud catch rate

#### 2.2 Model Architecture

**Model 1: Transaction Context Model (Gradient Boosting)**
- [ ] Features: 50-100 engineered features from transaction data
- [ ] Algorithm: XGBoost / LightGBM (trained offline, exported for z/OS inference)
- [ ] Rationale: Fast inference, handles tabular data well, interpretable feature importance

**Model 2: Behavioral Biometrics Anomaly Detector**
- [ ] Features: Per-session behavioral metrics compared to user's historical baseline
- [ ] Algorithm: Isolation Forest or autoencoder (unsupervised—detects deviation from user's normal behavior)
- [ ] Rationale: Catches account takeover even when credentials are correct

**Model 3: Graph-Based Fraud Ring Detection**
- [ ] Features: Graph embeddings from transaction network (sender-recipient graphs)
- [ ] Algorithm: Node2Vec or GraphSAGE embeddings + simple classifier
- [ ] Rationale: Detects organized fraud rings that individual transaction models miss
- [ ] Update frequency: Batch (daily/hourly embedding refresh)

**Ensemble:**
- [ ] Weighted combination of Model 1 + Model 2 scores
- [ ] Model 3 provides "ring risk" flag that boosts score when triggered
- [ ] Final output: risk_score (0-100) + explanation (top 3 contributing signals)

#### 2.3 Training Pipeline

**Offline Training (on IBM Z or connected platform):**
- [ ] Extract historical transaction data (with labels)
- [ ] Engineer features (same pipeline as real-time for consistency)
- [ ] Split: train/val/test by time (not random—avoid look-ahead bias)
- [ ] Train models with cross-validation
- [ ] Calibrate probability outputs (Platt scaling / isotonic regression)
- [ ] Export model artifacts (ONNX or native format for ML for z/OS)
- [ ] Validate latency benchmark on target hardware

**Key Considerations:**
- [ ] Handle extreme class imbalance (fraud is ~0.1% of transactions)
- [ ] Use time-based splits to prevent leakage
- [ ] Check for bias: ensure model doesn't unfairly flag specific demographics/geos
- [ ] Document model cards: intended use, limitations, performance by segment

#### 2.4 ML for z/OS Deployment Preparation
- [ ] Convert trained model to format supported by ML for z/OS
- [ ] Test inference latency on IBM Z hardware
- [ ] Validate model output matches offline predictions (numerical stability)
- [ ] Set up model versioning and rollback capability

---

### Phase 3: Real-Time Scoring Integration (Weeks 5-7)

#### 3.1 Transactional AI Integration

**Integration Pattern:**
```
Transaction arrives → Extract features → Call ML for z/OS scoring → Decision → Commit/Rollback
```

**Implementation Steps:**
- [x] Insert scoring call into transaction flow (CICS transaction, IMS application, or z/OS batch that feeds real-time)
- [x] Feature computation must be fast (pre-computed features where possible)
- [x] Scoring call: synchronous, timeout-bounded (fail-open or fail-closed per policy)
- [x] Decision engine applies business rules on top of ML score:
  - [x] score < 30: Allow
  - [x] 30-60: Allow with logging
  - [x] 60-80: Step-up authentication (OTP, biometric challenge)
  - [x] 80-95: Hold for review
  - [x] > 95: Block
- [x] Log every decision with: transaction ID, score, model version, explanation, decision, timestamp

#### 3.2 Behavioral Biometrics Collection

**Frontend Collection (lightweight SDK):**
- [x] Keystroke dynamics: keydown/keyup timing, digraphs/trigraphs patterns
- [x] Mouse dynamics: movement velocity, curvature, click positions
- [x] Touch dynamics (mobile): swipe pressure, speed, angle
- [x] Session-level aggregates sent with transaction

**Privacy Considerations:**
- [x] Collect only behavioral patterns, not content (what they type, not the text)
- [x] Hash user identity before sending behavioral data
- [x] Allow users to opt out (fall back to transaction-only scoring)

#### 3.3 Feature Store (Real-Time)

**Requirements:**
- [x] Low-latency reads (sub-millisecond) for real-time features
- [x] Stores: user's historical aggregates, behavioral baselines, device history
- [x] Implementation options on IBM Z:
  - [x] In-memory data cache (IMS DB, DB2 with caching, or attached Redis)
  - [x] Pre-computed feature tables updated by streaming pipeline

**Features to Precompute:**
- [x] User's avg transaction amount (7d, 30d rolling)
- [x] User's typical transaction locations
- [x] User's typical devices
- [x] User's transaction velocity (count/hr, count/day)
- [x] Time since last transaction
- [x] Account age
- [x] Historical fraud flag on account

---

### Phase 4: Decision Engine & UI (Weeks 6-8)

#### 4.1 Decision Engine

**Rules Layer (on top of ML scores):**
```
IF risk_score > 95: BLOCK
ELIF risk_score > 80:
    IF user has 2FA enrolled: REQUIRE_2FA
    ELSE: HOLD_FOR_REVIEW
ELIF risk_score > 60:
    IF behavioral_anomaly_detected: REQUIRE_2FA
    ELSE: ALLOW_WITH_LOGGING
ELIF graph_ring_flag == TRUE: BOOST_SCORE_BY_15
ELSE: ALLOW
```

**Configurable thresholds** per segment (new user vs. established, high-value vs. standard, etc.)
- [ ] Implement configurable threshold per segment

#### 4.2 Investigator Dashboard

**Purpose:** Allow fraud analysts to review flagged transactions, confirm/refute fraud, provide feedback loop.

**Features:**
- [ ] Queue of held/review transactions sorted by risk score
- [ ] Side-by-side: transaction details + behavioral biometrics visualization + model explanation
- [ ] One-click actions: confirm fraud, clear false positive, request more info
- [ ] Feedback captured → retraining data

**Technical:**
- [ ] Web UI (React/Angular) on z/OS via z/OS Connect or connected app server
- [ ] Data from IBM Z databases
- [ ] Audit trail of all investigator actions

#### 4.3 Alerting & Monitoring

- [ ] Real-time alert when fraud detection rate spikes or drops (model drift detection)
- [ ] Latency monitoring: p50/p95/p99 of scoring call
- [ ] False positive rate tracking (customer complaints, step-up abandonment)
- [ ] Model performance dashboard: score distribution, capture rate (when labels arrive)

---

### Phase 5: Security, Compliance & Responsible AI (Weeks 7-8)

#### 5.1 IBM Z Security Integration

- [ ] **Encryption:** All fraud data encrypted at rest (DB2 encryption, zFS encryption) and in transit (TLS)
- [ ] **Access Control:** RACF profiles control who can view fraud scores, investigator dashboards, model configurations
- [ ] **Audit Logging:** Every scoring decision, model version, threshold change logged to immutable audit trail
- [ ] **Secure Execution:** ML inference runs in z/OS secure enclave; model artifacts protected

#### 5.2 Responsible AI Controls

- [ ] **Explainability:** Every fraud score includes top contributing features (SHAP values or feature importance)
- [ ] **Bias Monitoring:** Track score distribution and false positive rate by protected attributes (where legally permissible)
- [ ] **Human-in-the-loop:** High-risk decisions (blocks) can require analyst confirmation for certain segments
- [ ] **Model Governance:** Versioned models, approval workflow for model updates, rollback capability
- [ ] **Privacy:** Behavioral biometrics processed without storing raw input data; data retained only as long as needed

#### 5.3 Compliance Considerations
- [ ] PCI DSS: If handling card data, ensure PCI compliance in architecture
- [ ] GDPR/CCPA: Right to explanation—model explanations support this
- [ ] Model risk management: Document model development, validation, and monitoring (SR 11-7 style if financial)

---

### Phase 6: Testing & Validation (Weeks 8-9)

#### 6.1 Unit & Integration Testing
- [ ] Feature computation: verify real-time features match offline features
- [ ] Scoring integration: test end-to-end transaction flow with scoring
- [ ] Decision engine: test all score ranges and rule combinations
- [ ] Latency: load test scoring path at target transaction volume

#### 6.2 Model Validation
- [ ] Holdout test set performance (from Phase 2)
- [ ] Backtest on historical fraud events: would this model have caught them?
- [ ] Adversarial testing: simulate fraud patterns, verify detection
- [ ] Stress test: behavior when feature data is missing/partial

#### 6.3 Operational Readiness
- [ ] Runbook: how to monitor, how to respond to alerts, how to roll back model
- [ ] Failover: what happens if ML for z/OS is unavailable (fallback to rules-only?)
- [ ] Capacity planning: sizing for peak transaction volume

---

### Phase 7: Deployment & Demo Preparation (Week 9-10)

#### 7.1 Deployment
- [ ] Deploy model to ML for z/OS production environment
- [ ] Enable scoring in transaction path (initially in shadow mode: score but don't block)
- [ ] Monitor shadow mode for 24-48 hours: score distribution, latency, errors
- [ ] Flip to active: start making decisions

#### 7.2 Datathon Demo Preparation

**Demo Storyline (3-5 minutes):**
- [ ] **Problem hook:** Show a fraudulent transaction slipping through a traditional system
- [ ] **Live demo:** Process a transaction through the system—show features being computed, model scoring in real-time, decision being made
- [ ] **Behavioral biometrics:** Show how the system detects account takeover even with correct credentials
- [ ] **Investigator view:** Show a held transaction in the dashboard with explanation
- [ ] **IBM Z value:** Emphasize latency, security, transactional integrity

**Demo Data:**
- [ ] Prepare realistic transaction scenarios (normal, suspicious, clear fraud)
- [ ] Pre-load behavioral profiles for demo users
- [ ] Have investigator dashboard ready with sample flagged transactions

**Key Messages for Judges:**
- [ ] "This runs inside the transaction—fraud is caught before money moves"
- [ ] "IBM Z's security means fraud data never leaves the secure perimeter"
- [ ] "ML for z/OS gives us sub-50ms inference at mainframe scale"
- [ ] "Every decision is explainable and auditable"

---

## 4. Technology Stack

| Component | Technology | IBM Z Relevance | Status |
|-----------|------------|-----------------|--------|
| Transaction processing | CICS, IMS/DC, or z/OS batch | Native IBM Z transaction systems | [ ] |
| Streaming | IBM Broker for z/OS / Kafka on z/OS | Real-time data on IBM Z | [ ] |
| ML training | Python (scikit-learn, XGBoost, PyTorch) on connected platform | Models exported to IBM Z | [ ] |
| ML inference | ML for z/OS | **Core IBM Z AI capability** | [ ] |
| Feature store | DB2, IMS DB, or in-memory cache on z/OS | Low-latency data access on IBM Z | [ ] |
| Behavioral collection | Lightweight JavaScript/mobile SDK | Front-end, not IBM Z-specific | [ ] |
| Decision engine | Custom logic in transaction flow | Integrated with IBM Z transactions | [ ] |
| Dashboard | Web app (React) via z/OS Connect | Exposes IBM Z data to web UI | [ ] |
| Security | RACF, CPACF, z/OS encryption | **IBM Z security capabilities** | [ ] |
| Audit logging | z/OS audit facilities, IMS/DC logs | Immutable audit trail on IBM Z | [ ] |

---

## 5. Data Requirements & Sourcing

### For Training
- [ ] Historical transactions (at least 6 months, ideally 12+)
- [ ] Fraud labels (confirmed fraud, chargebacks, customer reports)
- [ ] If real data unavailable: use public fraud datasets (Credit Card Fraud Detection - Kaggle, IEEE-CIS Fraud Detection) and adapt features to demonstrate architecture

### For Real-Time Scoring
- [ ] Live transaction feed (simulated for Datathon if needed)
- [ ] User behavioral profiles (build from demo data)
- [ ] Device/geo enrichment data

### Data Mocking Strategy (if no real data)
- [ ] Use public dataset as training base
- [ ] Build transaction simulator that generates realistic streaming transactions with embedded fraud patterns
- [ ] Simulator should produce: transaction events, behavioral events, labels (for evaluation)

---

## 6. Risks & Mitigations

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| No real transaction data | Can't train real model | Use public datasets + transaction simulator for demo; clearly state data limitation | [ ] |
| ML for z/OS access/_setup complexity | Cannot deploy inference on IBM Z | Have fallback: export model as ONNX, run inference in z/OS Java/Python if ML for z/OS unavailable; still demonstrate IBM Z integration | [ ] |
| Latency targets not met | Real-time claim undermined | Profile early; simplify model if needed; demonstrate latency benchmarking | [ ] |
| Behavioral biometrics too complex for timeline | Scope creep | Make biometrics optional/secondary; core demo can work with transaction features alone | [ ] |
| Class imbalance makes model hard to train | Poor model performance | Use appropriate techniques: class weighting, focal loss, oversampling; focus on precision@recall metric | [ ] |

---

## 7. Deliverables Checklist

### Technical Deliverables
- [ ] IBM Z environment configured with ML for z/OS
- [ ] Real-time feature computation pipeline
- [ ] Trained fraud detection model (exportable to z/OS)
- [ ] Model deployed for inference on IBM Z
- [ ] Transaction integration: scoring called in transaction path
- [ ] Decision engine with configurable thresholds
- [ ] Behavioral biometrics collection (SDK + feature extraction)
- [ ] Investigator dashboard (web UI)
- [ ] Monitoring/alerting setup
- [ ] Security controls: encryption, access control, audit logging

### Documentation Deliverables
- [ ] Architecture diagram
- [ ] Model card (training data, performance, limitations, intended use)
- [ ] API documentation (scoring API, feature API)
- [ ] Runbook (monitoring, troubleshooting, rollback)
- [ ] Demo script and scenario data

### Datathon Deliverables
- [ ] Working demo (live or recorded fallback)
- [ ] Presentation covering: problem, solution, IBM Z integration, results
- [ ] Clear explanation of AI + data methodology
- [ ] Responsible AI documentation

---

## 8. Success Criteria

| Criterion | Target | Verified |
|-----------|--------|----------|
| Inference latency (p99) | < 50ms | [ ] |
| Fraud capture rate (at demo) | Catch demonstrated fraud transactions | [ ] |
| False positive rate (demo) | Low enough to not block legitimate transactions in demo | [ ] |
| IBM Z integration depth | Scoring runs on IBM Z; data stays on IBM Z; audit logging on IBM Z | [ ] |
| Explainability | Every score has top 3 contributing features | [ ] |
| Demo polish | Smooth live demo or high-quality recorded demo | [ ] |

---
