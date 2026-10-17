"""Alias so both `python -m fraud_detection.train` and `.training` work."""

from fraud_detection.training import main

if __name__ == "__main__":
    raise SystemExit(main())