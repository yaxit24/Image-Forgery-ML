# src/ensemble.py
"""
Ensemble decision logic combining CNN and RandomForest predictions.

Weighted average approach:
  - CNN: 60% weight (better at visual pattern recognition)
  - RF:  40% weight (good at statistical feature anomalies)

When models disagree, the output explicitly flags uncertainty.
"""


def ensemble_predict(cnn_label, cnn_conf, rf_pred, rf_prob):
    """
    Combine CNN and RF predictions into a single verdict.

    Args:
        cnn_label: str, "forged" or "authentic"
        cnn_conf:  float, confidence in the CNN's own prediction [0,1]
        rf_pred:   int, 0=authentic, 1=forged
        rf_prob:   float, probability of class 1 (forged) from RF

    Returns:
        dict with keys:
            verdict     : str,   "forged" or "authentic"
            confidence  : float, ensemble confidence [0,1]
            agreement   : str,   "full" or "partial"
            cnn_label   : str
            cnn_conf    : float
            rf_label    : str
            rf_conf     : float
            details     : str,   human-readable explanation
    """
    CNN_WEIGHT = 0.6
    RF_WEIGHT  = 0.4

    # Normalize both to "probability of forged"
    cnn_forged_prob = cnn_conf if cnn_label == "forged" else (1.0 - cnn_conf)
    rf_forged_prob  = rf_prob

    # Weighted ensemble
    ensemble_forged_prob = CNN_WEIGHT * cnn_forged_prob + RF_WEIGHT * rf_forged_prob

    # Decision threshold
    if ensemble_forged_prob >= 0.5:
        verdict = "forged"
        confidence = ensemble_forged_prob
    else:
        verdict = "authentic"
        confidence = 1.0 - ensemble_forged_prob

    # Agreement check
    rf_label = "forged" if rf_pred == 1 else "authentic"
    models_agree = (cnn_label == rf_label)
    agreement = "full" if models_agree else "partial"

    # Human-readable explanation
    if models_agree:
        details = (
            f"Both models agree: {verdict} "
            f"(CNN: {cnn_conf:.0%}, RF: {rf_prob:.0%} forged probability)"
        )
    else:
        details = (
            f"Models disagree — CNN says {cnn_label} ({cnn_conf:.0%}), "
            f"RF says {rf_label} ({rf_prob:.0%}). "
            f"Ensemble leans {verdict} ({confidence:.0%} confidence). "
            f"Interpret with caution."
        )

    return {
        "verdict":    verdict,
        "confidence": round(confidence, 4),
        "agreement":  agreement,
        "cnn_label":  cnn_label,
        "cnn_conf":   round(cnn_conf, 4),
        "rf_label":   rf_label,
        "rf_conf":    round(rf_prob, 4),
        "details":    details,
    }
