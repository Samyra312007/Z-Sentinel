"""Real-time payment fraud detection with behavioral biometrics.

Phase 1 (Foundation & Data): synthetic transaction stream, biometric sample
generation, real-time feature computation, and a placeholder rules-only
decision engine used until the trained model from Phase 2 arrives.

Run the end-to-end demo:

    python -m fraud_detection --transactions 500 --seed 42
"""

__version__ = "0.1.0"