# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA) & PREPROCESSING PIPELINE
# European Social Survey (ESS) – Rounds 8, 9, 10, 11
# Target variable: vteurmmb (pro-EU voting intention)
# =============================================================================

# --- Libraries ---
import pandas as pd
import numpy as np
import miceforest as mf
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. DATA LOADING & CLEANING
# =============================================================================

def load_and_clean(filepath, source_name):
    """
    Loads an ESS round from Excel, standardises column names,
    and tags each row with its survey round for later traceability.
    """
    df = pd.read_excel(filepath, header=0)
    df.columns = df.columns.map(str).str.strip().str.lower()
    df["source"] = source_name
    return df

# Load all four ESS rounds; each will be harmonised before concatenation.
df_1 = load_and_clean("ESS11.xlsx", "ESS11")
df_2 = load_and_clean("ESS10.xlsx", "ESS10")
df_3 = load_and_clean("ESS9.xlsx",  "ESS9")
df_4 = load_and_clean("ESS8.xlsx",  "ESS8")


# =============================================================================
# 2. VARIABLE SELECTION
# =============================================================================

# Valid value ranges per variable; values outside these bounds are treated
# as missing (ESS uses codes such as 77, 88, 99 for "refusal / don't know").
column_ranges = {
    "vteurmmb": (1, 2),
    "stfeco":   (0, 10),
    "atchctr":  (0, 10),
    "euftf":    (0, 10),
    "atcherp":  (0, 10),
    "trstep":   (0, 10),
    "hincfel":  (1, 4),
    "agea":     (0, 120),
    "lrscale":  (0, 10),
    "trstplt":  (0, 10),
    "trstprt":  (0, 10),
    "trstprl":  (0, 10),
    "trstlgl":  (0, 10),
    "trstplc":  (0, 10),
    "stfdem":   (0, 10),
    "stfedu":   (0, 10),
    "stfhlth":  (0, 10),
    "stfgov":   (0, 10),
    "polintr":  (1, 9),
    "imueclt":  (0, 10),
    "imwbcnt":  (0, 10),
    "imbgeco":  (0, 10),
    "imsmetn":  (1, 4),
    "imdfetn":  (1, 4),
    "impcntr":  (1, 4),
    "edulvlb":  (0, 1000)
}

expected_cols = list(column_ranges.keys()) + ["idno", "source"]

def safe_subset(df):
    """
    Retains only columns present in both the dataframe and the expected list,
    avoiding KeyErrors when a variable is absent in a particular ESS round.
    """
    actual_cols = df.columns.intersection(expected_cols)
    return df[actual_cols].copy()

df_1_subset = safe_subset(df_1)
df_2_subset = safe_subset(df_2)
df_3_subset = safe_subset(df_3)
df_4_subset = safe_subset(df_4)

# Stack all rounds into a single dataframe; reset index for a clean row sequence.
df_subset = pd.concat(
    [df_1_subset, df_2_subset, df_3_subset, df_4_subset],
    axis=0, ignore_index=True
)


# =============================================================================
# 3. EDUCATION VARIABLE HARMONISATION (edulvlb)
# =============================================================================

# ESS encodes education as a 3-digit ISCED-based code (e.g. 313, 720).
# The leading digit captures the broad level (0–8), which is sufficient
# for modelling and ensures cross-round comparability.
if 'edulvlb' in df_subset.columns:
    df_subset['edulvlb'] = pd.to_numeric(df_subset['edulvlb'], errors='coerce')

    def get_first_digit(x):
        if pd.isna(x):
            return pd.NA
        if x in (0, 1):          # single-digit codes stay as-is
            return int(x)
        return int(str(int(x))[0])

    df_subset['edulvlb'] = df_subset['edulvlb'].apply(get_first_digit)
    column_ranges['edulvlb'] = (0, 8)   # update valid range after recoding

print(f"Total observations : {df_subset.shape[0]}")
print(f"Retained columns   : {df_subset.columns.tolist()}")
print(df_subset["source"].value_counts().to_frame("N"))


# =============================================================================
# 4. MISSING DATA ANALYSIS
# =============================================================================

# Out-of-range values are coded as missing by ESS convention;
# combining them with actual NaNs gives a unified missing-data count.
print("\nMissing Data Summary:\n")
for col, (low, high) in column_ranges.items():
    s       = df_subset[col]
    total   = len(s)
    missing = ((~s.between(low, high)) | s.isna()).sum()
    valid   = total - missing
    print(
        f"{col:8} | Total: {total:5} | Valid: {valid:5} "
        f"| Missing: {missing:5} | Missing %: {missing/total:5.1%}"
    )

# Replace out-of-range values with NaN so downstream steps treat them uniformly.
for col, (low, high) in column_ranges.items():
    df_subset.loc[~df_subset[col].between(low, high), col] = pd.NA

# Ensure all variables are numeric; coerce any remaining non-numeric entries to NaN.
for col in column_ranges:
    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')


# =============================================================================
# 5. MISSING DATA MECHANISM TEST (MCAR vs. MAR)
# =============================================================================

# Chi-square tests on missingness-indicator pairs help determine whether
# data are Missing Completely At Random (MCAR) or Missing At Random (MAR),
# which informs the choice of imputation strategy.
print("\nChi-Square Tests for Missingness Dependency:\n")
for var in column_ranges:
    df_subset[f"{var}_missing"] = df_subset[var].isna()

targets = ["vteurmmb", "lrscale"]
others  = [v for v in column_ranges if v not in targets]

for target in targets:
    print(f"\nIs '{target}_missing' associated with other variables?")
    for other in others + [v for v in targets if v != target]:
        contingency = pd.crosstab(
            df_subset[f"{target}_missing"],
            df_subset[f"{other}_missing"]
        )
        if contingency.shape == (2, 2):
            try:
                _, p, _, _ = chi2_contingency(contingency)
                print(f"  {other}_missing : p = {p:.4f}")
            except Exception as e:
                print(f"  {other}_missing : test failed – {e}")
        else:
            print(f"  {other}_missing : insufficient data for test")


# =============================================================================
# 6. MULTIPLE IMPUTATION (MICE)
# =============================================================================

# MICE (Multiple Imputation by Chained Equations) is applied because
# the MAR tests above indicate non-random missingness patterns,
# making single imputation methods (mean/median) inappropriate.
# NOTE: Imputation is performed on the full dataset before train/test split;
# ensure the fitted kernel is applied separately to each split in modelling.
kernel = mf.ImputationKernel(
    df_subset[list(column_ranges.keys())],
    random_state=42
)
kernel.mice(5)

df_imputed = kernel.complete_data(0)

print("\nImputed data – first 5 rows:")
print(df_imputed.head())

# Verify that imputation respects original value ranges (no negative values expected).
print("\nPost-imputation min/max check:")
for col in column_ranges:
    min_val = df_imputed[col].min()
    max_val = df_imputed[col].max()
    print(
        f"{col}: Min={min_val:.2f}, Max={max_val:.2f}, "
        f"Negative: {min_val < 0}"
    )


# =============================================================================
# 7. FEATURE ENGINEERING
# =============================================================================

print("\n--- Feature Engineering ---")
df_original = df_imputed.copy()   # preserve pre-engineering snapshot for reference
print("Available features:", df_imputed.columns.tolist())


# --- 7a. Direction Harmonisation ---
# Several Likert-scale items are reverse-coded (higher = more negative attitude).
# Reversing them ensures that higher values consistently indicate a more
# positive/permissive stance across all composite indices.
direction_reversal = {
    'imdfetn': True,   # 1=allow many → 4=allow none  (reversed to: 4=most open)
    'imsmetn': True,
    'impcntr': True,
    'hincfel': True,   # 1=living comfortably → 4=very difficult
    'polintr': True    # 1=very interested   → 4=not at all interested
}

for col, reverse in direction_reversal.items():
    if col in df_imputed.columns and reverse:
        lo, hi = column_ranges.get(col, (0, 10))
        df_imputed[col] = hi - df_imputed[col] + lo


# --- 7b. Institutional Trust Index ---
# Aggregating five institutional trust items into a single composite reduces
# dimensionality and noise while capturing the underlying trust construct.
trust_columns = [
    c for c in ['trstprt', 'trstplt', 'trstprl', 'trstlgl', 'trstplc']
    if c in df_imputed.columns
]

if trust_columns:
    df_imputed['institutional_trust'] = df_imputed[trust_columns].mean(axis=1)

    min_v, max_v = df_imputed['institutional_trust'].min(), df_imputed['institutional_trust'].max()
    df_imputed['institutional_trust_norm'] = (
        (df_imputed['institutional_trust'] - min_v) / (max_v - min_v)
    )
    print(df_imputed[trust_columns + ['institutional_trust', 'institutional_trust_norm']].head())


# --- 7c. Immigration Openness Index ---
# The three 1–4 scale items are first rescaled to 0–10 for compatibility
# with the remaining immigration items before averaging into a single index.
df_imputed['imdfetn_10'] = (df_imputed['imdfetn'] - 1) / 3 * 10
df_imputed['imsmetn_10'] = (df_imputed['imsmetn'] - 1) / 3 * 10
df_imputed['impcntr_10'] = (df_imputed['impcntr'] - 1) / 3 * 10

immigration_scaled = [
    'imdfetn_10', 'imsmetn_10', 'impcntr_10',
    'imwbcnt', 'imbgeco', 'imueclt'
]

df_imputed['immigration_openness'] = df_imputed[immigration_scaled].mean(axis=1)

min_v, max_v = df_imputed['immigration_openness'].min(), df_imputed['immigration_openness'].max()
df_imputed['immigration_openness_norm'] = (
    (df_imputed['immigration_openness'] - min_v) / (max_v - min_v)
)
print(df_imputed[immigration_scaled + ['immigration_openness', 'immigration_openness_norm']].head())


# --- 7d. National Governance Satisfaction Index ---
# Combining domain-specific satisfaction items captures overall government
# performance perception more reliably than any single survey item.
satisfaction_columns = [
    c for c in ['stfdem', 'stfeco', 'stfedu', 'stfhlth', 'stfgov']
    if c in df_imputed.columns
]

if satisfaction_columns:
    df_imputed['satisfaction_index'] = df_imputed[satisfaction_columns].mean(axis=1)

    min_v, max_v = df_imputed['satisfaction_index'].min(), df_imputed['satisfaction_index'].max()
    df_imputed['satisfaction_index_norm'] = (
        (df_imputed['satisfaction_index'] - min_v) / (max_v - min_v)
    )

print("\nSatisfaction index – first 5 rows:")
print(df_imputed[satisfaction_columns + ['satisfaction_index', 'satisfaction_index_norm']].head())


# --- 7e. EU Sentiment Index ---
# Three EU-specific items (federalism preference, EU attachment, EU parliament trust)
# are averaged to form a parsimonious measure of pro-EU orientation.
eu_columns = [c for c in ['euftf', 'atcherp', 'trstep'] if c in df_imputed.columns]

if eu_columns:
    df_imputed['eu_sentiment'] = df_imputed[eu_columns].mean(axis=1)

    min_v, max_v = df_imputed['eu_sentiment'].min(), df_imputed['eu_sentiment'].max()
    df_imputed['eu_sentiment_norm'] = (
        (df_imputed['eu_sentiment'] - min_v) / (max_v - min_v)
    )
    print(df_imputed[eu_columns + ['eu_sentiment', 'eu_sentiment_norm']].head())


# --- 7f. Age Group Segmentation ---
# Discretising age into generational cohorts allows the model to capture
# non-linear age effects without assuming a strictly linear relationship.
if 'agea' in df_imputed.columns:
    df_imputed['age_group'] = pd.cut(
        df_imputed['agea'],
        bins=[0, 25, 40, 55, 70, 120],
        labels=['0-25', '26-40', '41-55', '56-70', '71+'],
        right=False
    ).astype('category')


# --- 7g. Political Extremism (Distance from Centre) ---
# Absolute distance from the scale midpoint (5) quantifies ideological extremism
# independently of direction (left vs. right), which may have a distinct effect
# on EU voting behaviour.
if 'lrscale' in df_imputed.columns:
    df_imputed['distance_from_center'] = (df_imputed['lrscale'] - 5).abs()

    df_imputed['political_wing'] = pd.cut(
        df_imputed['lrscale'],
        bins=[0, 3, 7, 10],
        labels=['left', 'center', 'right'],
        include_lowest=True
    ).astype('category')


# =============================================================================
# 8. FEATURE SUMMARY
# =============================================================================

pd.set_option('display.max_rows', None)

def full_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns dtype, missing count/%, min, and max for every column."""
    records = []
    for col in df.columns:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        is_num = pd.api.types.is_numeric_dtype(s)
        records.append({
            "feature"       : col,
            "dtype"         : str(s.dtype),
            "missing_count" : s.isna().sum(),
            "missing_%"     : round(s.isna().mean() * 100, 2),
            "min"           : s.min() if is_num else None,
            "max"           : s.max() if is_num else None,
        })
    return pd.DataFrame(records)

print(full_column_summary(df_imputed).to_string(index=False))


# =============================================================================
# 9. CORRELATION ANALYSIS
# =============================================================================

# Pearson correlations with the target variable provide an initial ranking
# of predictor relevance and help flag redundant features before modelling.
target     = 'vteurmmb'
numeric_df = df_imputed.select_dtypes(include=['number', 'bool']).copy()

if target not in numeric_df.columns:
    raise ValueError(f"Target column '{target}' is not numeric or not found.")

corr_series = (
    numeric_df.drop(columns=[target])
              .apply(lambda col: col.corr(numeric_df[target]))
              .dropna()
)
corr_sorted = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)

print("\n--- All numeric features: correlation with target ---")
print(corr_sorted.to_string())
print("\n--- Top 20 correlations ---")
print(corr_sorted.head(20))


# =============================================================================
# 10. SPEARMAN CORRELATION MATRIX & MULTICOLLINEARITY CHECK
# =============================================================================

# Spearman's ρ is used instead of Pearson because several variables are
# ordinal; it does not assume linearity or normality of distributions.
# Age-derived columns are excluded as they are ordinal-encoded categories.
df_filtered = df_imputed.loc[:, ~df_imputed.columns.str.startswith('age')]
df_numeric  = df_filtered.select_dtypes(include=['number'])

corr_matrix = df_numeric.corr(method='spearman')

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Spearman Correlation Matrix")
plt.tight_layout()
plt.show()

# Flag strongly correlated feature pairs (|ρ| ≥ 0.6) that may cause
# multicollinearity issues in linear models.
threshold  = 0.6
high_corr  = corr_matrix[(corr_matrix.abs() >= threshold) & (corr_matrix.abs() < 1.0)]

print(f"\nHighly correlated feature pairs (|ρ| ≥ {threshold}):")
print(high_corr.dropna(how='all').dropna(axis=1, how='all'))
