# configs.py

# =====================
# TRAINING (ID only)
# =====================

TRAIN_ID = {
    "digit_range": list(range(0, 5)),   # digits 0–4
    "num_digits": 2,
    "motion": "free",
    "collision_mode": "bounce",
    "anomaly": None
}

TRAIN_ID_VERTICAL = {
    "digit_range": list(range(0, 10)), # digits are as per normal
    "num_digits": 2,
    "motion": "vertical",
    "collision_mode": "bounce",
    "anomaly": None
}

# =====================
# ID ANOMALIES (0–4)
# =====================

ID_APPEAR = {
    "digit_range": list(range(0, 5)),
    "num_digits": 2,          # 2 → 3 digits
    "motion": "free",
    "collision_mode": "bounce",
    "anomaly": "appear"
}

ID_DISAPPEAR = {
    "digit_range": list(range(0, 5)),
    "num_digits": 2,          # 2 → 1 digit
    "motion": "free",
    "collision_mode": "bounce",
    "anomaly": "disappear"
}

ID_STICK = {
    "digit_range": list(range(0, 5)),
    "num_digits": 2,
    "motion": "free",
    "collision_mode": "slide",   # sticking anomaly
    "anomaly": "stick"
}

# Vertical variants
ID_APPEAR_VERTICAL = {
    "digit_range": list(range(0, 10)),
    "num_digits": 2,
    "motion": "vertical",
    "collision_mode": "bounce",
    "anomaly": "appear"
}

ID_DISAPPEAR_VERTICAL = {
    "digit_range": list(range(0, 10)),
    "num_digits": 2,
    "motion": "vertical",
    "collision_mode": "bounce",
    "anomaly": "disappear"
}

ID_STICK_VERTICAL = {
    "digit_range": list(range(0, 10)),
    "num_digits": 2,
    "motion": "vertical",
    "collision_mode": "slide",
    "anomaly": "stick"
}

# =====================
# OOD ANOMALIES (5–9)
# =====================

OOD_APPEAR = {
    "digit_range": list(range(5, 10)),
    "num_digits": 2,
    "motion": "free",
    "collision_mode": "bounce",
    "anomaly": "appear"
}

OOD_DISAPPEAR = {
    "digit_range": list(range(5, 10)),
    "num_digits": 2,
    "motion": "free",
    "collision_mode": "bounce",
    "anomaly": "disappear"
}

OOD_STICK = {
    "digit_range": list(range(5, 10)),
    "num_digits": 2,
    "motion": "free",
    "collision_mode": "slide",
    "anomaly": "stick"
}

# Horizontal bouncing variants
OOD_APPEAR_HORIZONTAL = {
    "digit_range": list(range(0, 10)),
    "num_digits": 2,
    "motion": "horizontal",
    "collision_mode": "bounce",
    "anomaly": "appear"
}

OOD_DISAPPEAR_HORIZONTAL = {
    "digit_range": list(range(0, 10)),
    "num_digits": 2,
    "motion": "horizontal",
    "collision_mode": "bounce",
    "anomaly": "disappear"
}

OOD_STICK_HORIZONTAL = {
    "digit_range": list(range(0, 10)),
    "num_digits": 2,
    "motion": "horizontal",
    "collision_mode": "slide",
    "anomaly": "stick"
}

# =====================
# NO ANOMALY (TEST)
# =====================

ID_NORMAL = {
    "digit_range": list(range(0, 5)),
    "num_digits": 2,
    "motion": "free",
    "collision_mode": "bounce",
    "anomaly": None
}

ID_NORMAL_VERTICAL = {
    "digit_range": list(range(0, 10)),
    "num_digits": 2,
    "motion": "vertical",
    "collision_mode": "bounce",
    "anomaly": None
}

OOD_NORMAL = {
    "digit_range": list(range(5, 10)),
    "num_digits": 2,
    "motion": "free",
    "collision_mode": "bounce",
    "anomaly": None
}

OOD_NORMAL_HORIZONTAL = {
    "digit_range": list(range(0, 10)),
    "num_digits": 2,
    "motion": "horizontal",
    "collision_mode": "bounce",
    "anomaly": None
}