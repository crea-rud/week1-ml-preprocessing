"""
generate_dataset.py
-------------------
Creates raw_dataset.csv  ->  a SYNTHETIC (computer-generated) student-performance dataset.

IMPORTANT: This data is NOT real. It is generated with NumPy using a fixed random seed,
so anyone who runs this script gets exactly the same file.
Real-world data-quality problems (missing values, duplicates, inconsistent text,
impossible values, numbers stored as text) are deliberately injected so the
Week 1 cleaning steps have something realistic to fix.

Run from the project root:   python scripts/generate_dataset.py
"""
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
N = 500  # number of unique students before duplicates are added

# ---------------------------------------------------------------- 1. clean base data
df = pd.DataFrame({
    "student_id": [f"S{i:04d}" for i in range(1, N + 1)],
    "age": rng.integers(17, 25, N),
    "gender": rng.choice(["Male", "Female", "Other"], N, p=[0.48, 0.48, 0.04]),
    "department": rng.choice(
        ["Computer Science", "Electronics", "Mechanical", "Commerce", "Arts"],
        N, p=[0.30, 0.20, 0.15, 0.20, 0.15]),
    "study_hours_per_week": np.clip(rng.normal(15, 6, N), 1, 40).round(1),
    "attendance_percent": np.clip(rng.normal(80, 12, N), 35, 100).round(1),
    "sleep_hours": np.clip(rng.normal(6.8, 1.1, N), 4, 10).round(1),
    "previous_score": np.clip(rng.normal(65, 14, N), 20, 100).round(1),
    "internet_access": rng.choice(["Yes", "No"], N, p=[0.85, 0.15]),
    "parental_education": rng.choice(
        ["High School", "Bachelor", "Master", "PhD"], N, p=[0.35, 0.40, 0.20, 0.05]),
    "extracurricular": rng.choice(["Yes", "No"], N, p=[0.45, 0.55]),
})

# Target column: built from a simple formula + random noise (so patterns are known by design)
internet_num = (df["internet_access"] == "Yes").astype(int)
df["final_score"] = np.clip(
    10 + 0.45 * df["previous_score"] + 0.9 * df["study_hours_per_week"]
    + 0.12 * df["attendance_percent"] + 0.8 * (df["sleep_hours"] - 6)
    + 2.0 * internet_num + rng.normal(0, 5, N), 0, 100).round(1)

# ---------------------------------------------------------------- 2. inject data-quality issues
# (a) attendance stored as text with a % sign in some rows
pct_rows = rng.choice(N, 40, replace=False)
df["attendance_percent"] = df["attendance_percent"].astype(object)
for i in pct_rows:
    df.at[i, "attendance_percent"] = f"{df.at[i, 'attendance_percent']}%"

# (b) inconsistent text (case, spaces, short forms)
def messy(col, mapping, n):
    idx = rng.choice(N, n, replace=False)
    for i in idx:
        df.at[i, col] = mapping(df.at[i, col])

messy("gender", lambda v: {"Male": rng.choice(["male", "M", " Male"]),
                           "Female": rng.choice(["FEMALE", "F", "female "]),
                           "Other": "other"}[v], 60)
messy("department", lambda v: {"Computer Science": rng.choice(["computer science", "CS", " Computer Science"]),
                               "Electronics": "electronics",
                               "Mechanical": "MECHANICAL ",
                               "Commerce": "commerce",
                               "Arts": "arts"}[v], 50)
messy("internet_access", lambda v: {"Yes": rng.choice(["yes", "Y"]), "No": rng.choice(["no", "N"])}[v], 40)
messy("parental_education", lambda v: v.lower(), 30)
messy("extracurricular", lambda v: v.lower(), 20)

# (c) impossible values (typing mistakes / sensor errors)
bad = rng.choice(N, 12, replace=False)
df.loc[bad[0:3], "age"] = [0, 99, -3]
df.loc[bad[3:6], "study_hours_per_week"] = [168, 120, 95]
df.loc[bad[6:9], "attendance_percent"] = [130, 150, -5]
df.loc[bad[9:11], "sleep_hours"] = [25, 0]
df.loc[bad[11], "previous_score"] = 250

# (d) missing values (about 3-6 % per column; a few missing target values)
for col, frac in [("age", 0.04), ("gender", 0.03), ("study_hours_per_week", 0.06),
                  ("attendance_percent", 0.05), ("sleep_hours", 0.05),
                  ("previous_score", 0.04), ("internet_access", 0.03),
                  ("parental_education", 0.05), ("final_score", 0.016)]:
    idx = rng.choice(N, int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

# (e) exact duplicate rows
dups = df.sample(25, random_state=SEED)
df = pd.concat([df, dups], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle rows

df.to_csv("data/raw_dataset.csv", index=False)
print("raw_dataset.csv written:", df.shape)
