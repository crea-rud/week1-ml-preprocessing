# Week 1 Project – Python for Machine Learning & Data Preprocessing

**Internship:** AI Pioneers Internship (Machine Learning) – Yuva Intern
**Topics covered:** Python fundamentals, NumPy, Pandas, data loading, cleaning, missing-value handling,
duplicate removal, feature selection, categorical encoding, numerical normalization, and EDA.

## ⚠️ Dataset disclosure
The dataset is **SYNTHETIC (computer-generated)**. It is **not real data** and was **not** taken from a public source.
`scripts/generate_dataset.py` creates it with NumPy (random seed 42), so it is exactly reproducible.
Data-quality problems (missing values, duplicate rows, inconsistent text, impossible values, numbers stored as text)
were **deliberately injected**. Patterns in the data (for example study hours vs. final score) were built into the
generator, so they are **not** real-world findings. **No machine-learning model is trained in Week 1, so no accuracy is reported.**

## Folder structure
```
week1_ml_preprocessing/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_dataset.csv             # synthetic raw data (525 rows, 12 columns, with problems)
│   ├── cleaned_dataset.csv         # cleaned, human-readable data (492 rows, 12 columns)
│   ├── encoded_scaled_dataset.csv  # all features encoded + min-max scaled (492 rows, 15 columns)
│   └── ml_ready_dataset.csv        # only the selected features + target (492 rows, 4 columns)
├── notebooks/
│   └── week1_preprocessing.ipynb   # main notebook (code, explanations, outputs, charts)
├── scripts/
│   └── generate_dataset.py         # recreates data/raw_dataset.csv
├── figures/                        # charts saved by the notebook (fig01 ... fig08)
├── outputs/
│   ├── cleaning_log.csv            # counts of what each cleaning step changed
│   └── report_stats.json           # numbers used in the report
└── report/
    ├── Week1_Project_Report.docx
    └── Week1_Project_Report.pdf
```

## Dependencies
Python 3.9 or newer. Install everything with:
```
pip install -r requirements.txt
```
Tested with Python 3.12, NumPy 2.5, Pandas 3.0, Matplotlib 3.11, Seaborn 0.13 and SciPy 1.18.

## How to run
1. Open a terminal in the project folder (`week1_ml_preprocessing`).
2. *(Optional)* Recreate the raw data: `python scripts/generate_dataset.py`
3. Start Jupyter: `jupyter notebook`
4. Open `notebooks/week1_preprocessing.ipynb`.
5. Choose **Kernel → Restart & Run All**. The notebook must be opened from the `notebooks/` folder because it uses relative paths such as `../data/raw_dataset.csv`.
6. After the run, `data/cleaned_dataset.csv`, `data/encoded_scaled_dataset.csv`, `data/ml_ready_dataset.csv`, the `figures/` charts and the `outputs/` files are re-created.

## What the notebook does
1. Python fundamentals and NumPy basics (short demos)
2. Load and inspect the raw data (shape, types, summary, missing values, duplicates, messy categories)
3. Clean: remove duplicates → standardise text → fix text-number column → flag impossible values → handle missing values (median / mode, drop rows without target) → cap outliers (IQR rule)
4. Export `cleaned_dataset.csv` and verify it by reading it back
5. Feature selection: drop identifier, correlation filter for numeric columns, ANOVA p-value filter for categorical columns
6. Categorical encoding: binary mapping, ordinal encoding, one-hot encoding
7. Normalization: min-max scaling (and z-score for comparison)
8. EDA: statistics, distributions, category counts, scatter plots, box plots, correlation heatmap
9. Export model-ready files and run automatic checks (no missing values, no duplicates, valid ranges, numeric types)

## Limitations
- Synthetic data: results describe the generator, not real students.
- Median/mode imputation and IQR capping are simple beginner methods.
- Feature-selection filters are simple and can miss weak signals.
