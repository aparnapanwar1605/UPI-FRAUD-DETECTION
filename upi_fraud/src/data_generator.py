"""
Synthetic UPI transaction generator.

Real UPI fraud data is private, so this module simulates it. Legitimate
behaviour and three fraud "playbooks" are modelled, with deliberate overlap and
noise so the problem is NOT trivially separable:

  1. Collect-request / social-engineering scam  (victim approves a fake request)
  2. Account takeover                           (new device, odd hour, burst)
  3. Mule / cash-out                            (fresh accounts, rapid pass-through)

The fraud rate is controlled (default 0.1%) so imbalance handling can be studied.
"""
import numpy as np
import pandas as pd

TXN_TYPES = ["P2P", "P2M", "BILL", "RECHARGE", "COLLECT"]
MERCHANT_CATS = ["none", "grocery", "food", "travel", "utilities", "electronics", "gaming", "other"]


def generate_upi_transactions(n=400_000, fraud_rate=0.001, days=90, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n * fraud_rate))
    n_legit = n - n_fraud

    # ------------------------------------------------------------------ legit
    legit = pd.DataFrame({
        "txn_type": rng.choice(TXN_TYPES, n_legit, p=[0.38, 0.38, 0.12, 0.08, 0.04]),
        "hour": np.clip(rng.normal(15, 5, n_legit), 0, 23).astype(int),
        "sender_account_age_days": rng.gamma(2.2, 400, n_legit).astype(int) + 5,
        "device_changed": rng.random(n_legit) < 0.015,
        "new_payee": rng.random(n_legit) < 0.18,
        "txns_last_1h": rng.poisson(0.25, n_legit),
        "failed_pin_attempts": rng.choice([0, 1, 2], n_legit, p=[0.965, 0.03, 0.005]),
        "location_mismatch": rng.random(n_legit) < 0.04,
        "receiver_account_age_days": rng.gamma(2.2, 400, n_legit).astype(int) + 5,
        "is_fraud": 0,
    })
    base = rng.lognormal(6.0, 1.1, n_legit)                        # median ~Rs 400
    legit["amount"] = np.round(np.clip(base, 1, 100_000), 2)
    legit["sender_avg_amount"] = np.round(np.clip(base * rng.lognormal(0, 0.5, n_legit), 20, 50_000), 2)

    # ------------------------------------------------------------------ fraud
    kinds = rng.choice(["collect_scam", "takeover", "mule"], n_fraud, p=[0.45, 0.35, 0.20])
    rows = []
    for k in kinds:
        if k == "collect_scam":
            r = dict(txn_type=rng.choice(["COLLECT", "P2P"], p=[0.55, 0.45]),
                     hour=int(np.clip(rng.normal(14, 6), 0, 23)),
                     sender_account_age_days=int(rng.gamma(2.2, 400)) + 5,
                     device_changed=rng.random() < 0.05,
                     new_payee=rng.random() < 0.92,
                     txns_last_1h=int(rng.poisson(0.4)),
                     failed_pin_attempts=int(rng.choice([0, 1], p=[0.9, 0.1])),
                     location_mismatch=rng.random() < 0.25,
                     receiver_account_age_days=int(rng.gamma(1.2, 90)) + 3,
                     amount=float(np.clip(rng.lognormal(8.3, 0.8), 200, 100_000)))
        elif k == "takeover":
            r = dict(txn_type=rng.choice(["P2P", "P2M"], p=[0.7, 0.3]),
                     hour=int(rng.choice([0, 1, 2, 3, 4, 23, 5, 22], p=[.2, .2, .15, .15, .1, .08, .07, .05])),
                     sender_account_age_days=int(rng.gamma(2.2, 400)) + 5,
                     device_changed=rng.random() < 0.85,
                     new_payee=rng.random() < 0.8,
                     txns_last_1h=int(rng.poisson(2.5)),
                     failed_pin_attempts=int(rng.choice([0, 1, 2, 3], p=[0.35, 0.3, 0.2, 0.15])),
                     location_mismatch=rng.random() < 0.7,
                     receiver_account_age_days=int(rng.gamma(1.5, 150)) + 3,
                     amount=float(np.clip(rng.lognormal(8.8, 0.7), 500, 100_000)))
        else:  # mule
            r = dict(txn_type=rng.choice(["P2P", "P2M"], p=[0.85, 0.15]),
                     hour=int(np.clip(rng.normal(13, 7), 0, 23)),
                     sender_account_age_days=int(rng.gamma(1.1, 60)) + 2,
                     device_changed=rng.random() < 0.1,
                     new_payee=rng.random() < 0.6,
                     txns_last_1h=int(rng.poisson(3.5)),
                     failed_pin_attempts=int(rng.choice([0, 1], p=[0.93, 0.07])),
                     location_mismatch=rng.random() < 0.15,
                     receiver_account_age_days=int(rng.gamma(1.1, 60)) + 2,
                     amount=float(np.clip(rng.lognormal(7.4, 0.9), 100, 60_000)))
        r["sender_avg_amount"] = float(np.clip(r["amount"] * rng.lognormal(-0.2, 0.9), 20, 50_000))
        r["is_fraud"] = 1
        r["_kind"] = k
        rows.append(r)
    fraud = pd.DataFrame(rows)

    # ------------------------------------------------- make the task realistic
    # (a) "stealth" fraud: ~35% of fraud mimics normal behaviour, leaving only weak signals
    stealth = fraud.sample(frac=0.30, random_state=seed).index
    fraud["_stealth"] = 0
    fraud.loc[stealth, "_stealth"] = 1
    k = len(stealth)
    fraud.loc[stealth, "hour"] = np.clip(rng.normal(15, 5, k), 0, 23).astype(int)
    fraud.loc[stealth, "device_changed"] = rng.random(k) < 0.03
    fraud.loc[stealth, "location_mismatch"] = rng.random(k) < 0.06
    fraud.loc[stealth, "failed_pin_attempts"] = 0
    fraud.loc[stealth, "txns_last_1h"] = rng.poisson(0.4, k)
    fraud.loc[stealth, "sender_account_age_days"] = rng.gamma(2.2, 400, k).astype(int) + 5
    fraud.loc[stealth, "receiver_account_age_days"] = rng.gamma(2.2, 400, k).astype(int) + 5
    fraud.loc[stealth, "amount"] = np.clip(rng.lognormal(6.6, 1.0, k), 50, 100_000)
    fraud.loc[stealth, "sender_avg_amount"] = fraud.loc[stealth, "amount"] * rng.lognormal(0, 0.5, k)

    # (b) "hard negatives": ~3% of legit look suspicious (travelling, new phone, big purchase)
    hn = rng.random(n_legit) < 0.018
    h = int(hn.sum())
    legit["_hard_negative"] = hn.astype(int)
    legit.loc[hn, "device_changed"] = rng.random(h) < 0.45
    legit.loc[hn, "new_payee"] = rng.random(h) < 0.7
    legit.loc[hn, "location_mismatch"] = rng.random(h) < 0.4
    legit.loc[hn, "hour"] = rng.choice([22, 23, 0, 1, 2, 3, 14, 15], h)
    legit.loc[hn, "txns_last_1h"] = rng.poisson(1.5, h)
    legit.loc[hn, "failed_pin_attempts"] = rng.choice([0, 1, 2], h, p=[0.7, 0.2, 0.1])
    legit.loc[hn, "receiver_account_age_days"] = rng.gamma(1.4, 150, h).astype(int) + 3
    legit.loc[hn, "amount"] = np.round(np.clip(rng.lognormal(7.9, 1.0, h), 100, 100_000), 2)
    legit.loc[hn, "sender_avg_amount"] = np.round(legit.loc[hn, "amount"] * rng.lognormal(0.3, 0.7, h), 2)

    # --------------------------------------------------------------- combine
    legit["_kind"] = "legit"
    legit["_stealth"] = 0
    fraud["_hard_negative"] = 0
    df = pd.concat([legit, fraud], ignore_index=True)
    df["amount"] = df["amount"].round(2)
    df["sender_avg_amount"] = df["sender_avg_amount"].round(2)
    df["amount_vs_avg_ratio"] = (df["amount"] / df["sender_avg_amount"]).round(3)
    df["merchant_category"] = np.where(
        df["txn_type"].isin(["P2P", "COLLECT"]), "none",
        rng.choice(MERCHANT_CATS[1:], len(df), p=[.25, .2, .1, .15, .1, .1, .1]))
    # fraud on merchants skews to gaming/electronics (cash-out via resellable goods)
    m = (df["is_fraud"] == 1) & (df["txn_type"] == "P2M")
    df.loc[m, "merchant_category"] = rng.choice(["gaming", "electronics", "other"], m.sum())

    # timestamps: random day in the window, clock time taken from the generated hour
    day = rng.integers(0, days, len(df))
    sec = rng.integers(0, 3600, len(df))
    df["timestamp"] = (pd.Timestamp("2025-01-01") + pd.to_timedelta(day, unit="D")
                       + pd.to_timedelta(df["hour"].values, unit="h") + pd.to_timedelta(sec, unit="s"))
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["txn_id"] = ["TXN" + str(i).zfill(7) for i in range(len(df))]

    # label noise-free, but make 'new_payee'-style flags ints for modelling
    for c in ["device_changed", "new_payee", "location_mismatch"]:
        df[c] = df[c].astype(int)
    cols = ["txn_id", "timestamp", "txn_type", "merchant_category", "amount", "sender_avg_amount",
            "amount_vs_avg_ratio", "hour", "sender_account_age_days", "receiver_account_age_days",
            "device_changed", "new_payee", "location_mismatch", "txns_last_1h",
            "failed_pin_attempts", "is_fraud", "_kind", "_stealth", "_hard_negative"]
    return df[cols]


if __name__ == "__main__":
    d = generate_upi_transactions()
    print(d.shape, "fraud rate:", d.is_fraud.mean())
    print(d.groupby("_kind").size())
