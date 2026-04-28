# decision_engine.py
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class StructuralBreak:
    year_split: int


def detect_structural_break(years, values):

    if len(values) < 8:
        return None

    best_score = 0
    best_year = None

    for i in range(3, len(values) - 3):
        left = values[:i]
        right = values[i:]

        slope1 = np.polyfit(range(len(left)), left, 1)[0]
        slope2 = np.polyfit(range(len(right)), right, 1)[0]

        score = abs(slope1 - slope2)

        if score > best_score:
            best_score = score
            best_year = years[i]

    if best_score > 1:
        return StructuralBreak(int(best_year))

    return None


def detect_abnormal_volatility(years, values):
    df = pd.DataFrame({"annee": years, "val": values})
    df["std"] = df["val"].rolling(4).std()
    threshold = df["std"].mean() + 2 * df["std"].std()
    return df[df["std"] > threshold]


def identify_turning_years(years, values):
    df = pd.DataFrame({"annee": years, "val": values})
    df["smooth"] = df["val"].rolling(3, center=True).mean()
    out = []
    for i in range(1, len(df) - 1):
        if df["smooth"].iloc[i] > df["smooth"].iloc[i - 1] and df["smooth"].iloc[i] > df["smooth"].iloc[i + 1]:
            out.append(df.iloc[i])
    return pd.DataFrame(out)


def analyze_cycles(years, values):
    if len(values) < 10:
        return {"dominant_period_years": np.nan}
    fft = np.abs(np.fft.rfft(values - np.mean(values)))
    freq = np.fft.rfftfreq(len(values), 1)
    idx = np.argmax(fft[1:]) + 1
    period = 1 / freq[idx] if freq[idx] > 0 else np.nan
    return {"dominant_period_years": period}