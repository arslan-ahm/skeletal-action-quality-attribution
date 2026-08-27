"""Reference-free, calibrated skeletal action-quality regression with validated
per-joint, per-phase attribution.

The one-sentence difference from the reference approach: instead of a single
DTW similarity number against a reference performance, this package predicts a
*calibrated* quality score with a predictive interval, and localises the quality
loss to specific joints and movement phases -- then measures whether that
localisation is actually correct against exact ground truth.
"""

__version__ = "1.0.0"
