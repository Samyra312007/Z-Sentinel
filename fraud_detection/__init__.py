"""Real-time payment fraud detection with behavioral biometrics.

Phase 1 (Foundation & Data): synthetic transaction stream, biometric sample
generation, real-time feature computation, graph ring features, and a
rules-only decision engine (fallback scoring).

Phase 2 (ML Model Development): pure-Python gradient-boosted trees on
transaction context, calibrated probabilities, behavioral + graph ensemble,
top-3 explanations, model artifact and model card.

Train and score:

    python -m fraud_detection.train --out models/fraud_model.json
    python -m fraud_detection --model models/fraud_model.json --transactions 2000 \
        --users 200 --fraud-rate 0.03 --warmup 2000

Rules-only fallback (no model needed):

    python -m fraud_detection --transactions 500 --seed 42

Tests:

    python -m unittest discover -s tests -v
"""

__version__ = "0.2.0"