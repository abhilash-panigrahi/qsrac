def validate_attributes(context: dict) -> bool:
    try:
        sensitivity = float(context.get("sensitivity_level", 0))
        device_trust = float(context.get("device_trust_score", 0))
    except (TypeError, ValueError):
        return False

    return sensitivity <= 5 and device_trust >= 0.2