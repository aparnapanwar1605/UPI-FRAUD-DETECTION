"""Feature engineering for the UPI fraud project."""
import numpy as np
import pandas as pd

NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4, 5]

RAW_NUM = [
    "amount_vs_avg_ratio", "hour", "sender_account_age_days", "receiver_account_age_days",
    "device_changed", "new_payee", "location_mismatch", "txns_last_1h", "failed_pin_attempts",
]
ENGINEERED_NUM = [
    "log_amount", "night", "recv_is_new", "sender_is_new", "burst", "big_vs_avg",
    "newdev_x_newpayee", "night_x_newpayee", "newpayee_x_newrecv", "newpayee_x_big", "risk_flags",
]
NUM_FEATURES = RAW_NUM + ENGINEERED_NUM
CAT_FEATURES = ["txn_type", "merchant_category"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural flags and interaction terms. Uses only information
    available at the moment of the transaction (no label / future leakage)."""
    d = df.copy()
    d["log_amount"] = np.log1p(d["amount"])
    d["night"] = d["hour"].isin(NIGHT_HOURS).astype(int)
    d["recv_is_new"] = (d["receiver_account_age_days"] < 60).astype(int)
    d["sender_is_new"] = (d["sender_account_age_days"] < 60).astype(int)
    d["burst"] = (d["txns_last_1h"] >= 3).astype(int)
    d["big_vs_avg"] = (d["amount_vs_avg_ratio"] > 3).astype(int)
    # interactions: fraud usually shows up as a *combination* of weak signals
    d["newdev_x_newpayee"] = d["device_changed"] * d["new_payee"]
    d["night_x_newpayee"] = d["night"] * d["new_payee"]
    d["newpayee_x_newrecv"] = d["new_payee"] * d["recv_is_new"]
    d["newpayee_x_big"] = d["new_payee"] * d["big_vs_avg"]
    d["risk_flags"] = (d["device_changed"] + d["new_payee"] + d["location_mismatch"] + d["night"]
                       + d["recv_is_new"] + d["burst"] + d["big_vs_avg"] + (d["failed_pin_attempts"] > 0))
    return d
