"""Real-time payment fraud detection with behavioral biometrics.

Phase 1 (Foundation & Data): synthetic transaction stream, biometric sample
generation, real-time feature computation, graph ring features, and a
rules-only decision engine (fallback scoring).

Phase 2 (ML Model Development): pure-Python gradient-boosted trees on
transaction context, calibrated probabilities, behavioral + graph ensemble,
top-3 explanations, model artifact and model card.

Phase 3 (Real-Time Scoring Integration): online feature store with precomputed
per-user aggregates, privacy-preserving behavioral-biometrics SDK and payload
ingestion, synchronous timeout-bounded scoring with fail-open/fail-closed
policy, settlement outcomes, and an append-only decision audit trail.

Train and score:

    python -m fraud_detection.train --out models/fraud_model.json
    python -m fraud_detection --model models/fraud_model.json --transactions 2000 \
        --users 200 --fraud-rate 0.03 --warmup 2000

Phase 3 end-to-end integration demo:

    ./run_phase3.sh
    python -m fraud_detection.realtime --model models/fraud_model.json \
        --transactions 500 --audit-log logs/phase3-decisions.jsonl

Rules-only fallback (no model needed):

    python -m fraud_detection --transactions 500 --seed 42

Tests:

    python -m unittest discover -s tests -v
"""

__version__ = "0.3.0"
