##################################
# --- Libraries ---
import pandas as pd
import numpy as np
import miceforest as mf
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cleaning function for datasets ---
def load_and_clean(filepath, source_name):
    df = pd.read_excel(filepath, header=0)
    df.columns = df.columns.map(str).str.strip().str.lower()
    df["source"] = source_name
    return df

# --- Reading files ---
df_1 = load_and_clean("ESS11.xlsx", "ESS11")
df_2 = load_and_clean("ESS10.xlsx", "ESS10")
df_3 = load_and_clean("ESS9.xlsx",  "ESS9")
df_4 = load_and_clean("ESS8.xlsx",  "ESS8")

# --- Columns ---
column_ranges = {
    "vteurmmb": (1, 2),
    "stfeco": (0, 10),
    "atchctr": (0, 10),
    "euftf": (0, 10),
    "atcherp": (0, 10),
    "trstep": (0, 10),
    "hincfel": (1, 4),
    "agea": (0, 120),
    "lrscale": (0, 10),
    "trstplt": (0, 10),
    "trstprt": (0, 10),
    "trstprl": (0, 10),
    "trstlgl": (0, 10),
    "trstplc": (0, 10),
    "stfdem": (0, 10),
    "stfedu": (0, 10),
    "stfhlth": (0, 10),
    "stfgov": (0, 10),
    "polintr": (1, 9),
    "imueclt": (0, 10),
    "imwbcnt": (0, 10),
    "imbgeco": (0, 10),
    "imsmetn": (1, 4),
    "imdfetn": (1, 4),
    "impcntr": (1, 4),
    "edulvlb": (0, 1000)
}

expected_cols = list(column_ranges.keys()) + ["idno", "source"]

# --- Sadece gerçekten olan sütunlarla alt setleri al ---
def safe_subset(df):
    actual_cols = df.columns.intersection(expected_cols)
    return df[actual_cols].copy()

df_1_subset = safe_subset(df_1)
df_2_subset = safe_subset(df_2)
df_3_subset = safe_subset(df_3)
df_4_subset = safe_subset(df_4)

# --- Concat datasets ---
df_subset = pd.concat([df_1_subset, df_2_subset, df_3_subset, df_4_subset], axis=0, ignore_index=True)


#####################################################################
# --- DEĞIŞIM 1: edulvlb kolonunun ilk hanesine göre düzenlenmesi ---
#####################################################################
if 'edulvlb' in df_subset.columns:
    # İlk önce edulvlb kolonunu sayısal tipe dönüştürelim
    df_subset['edulvlb'] = pd.to_numeric(df_subset['edulvlb'], errors='coerce')
    
    # Fonksiyon: ilk haneyi almak için
    def get_first_digit(x):
        if pd.isna(x):
            return pd.NA
        # 0 için özel durum, 0 kalacak
        if x == 0:
            return 0
        # 1 için özel durum, 1 kalacak
        if x == 1:
            return 1
        # Diğerleri için ilk haneyi al
        return int(str(int(x))[0])
    
    # İlk haneyi al
    df_subset['edulvlb'] = df_subset['edulvlb'].apply(get_first_digit)
    
    # edulvlb için range'i güncelle
    column_ranges['edulvlb'] = (0, 8)

print(f"🔢 Total observation: {df_subset.shape[0]}")
print(f"🧩 Ortak Sütunlar: {df_subset.columns.tolist()}")
print(df_subset["source"].value_counts().to_frame("Gözlem Sayısı"))




#####################################################################
# ----- MISSING DATA IMPUTATION ------ # (Invalid + missing = Missing data)
#####################################################################

print("Missing Data Table:\n")
for col, (low, high) in column_ranges.items():
    s = df_subset[col]
    total = len(s)
    missing = (~s.between(low, high)) | (s.isna())  # out of range & NaN 
    missing_sum = missing.sum()
    valid = total - missing_sum
    print(f"{col:8} | Total: {total:5} | Valid: {valid:5} | Missing: {missing_sum:5} | Missing %: {missing_sum/total:5.1%}")

# --- Converting invalids to NaN ---
for col, (low, high) in column_ranges.items():
    df_subset.loc[~df_subset[col].between(low, high), col] = pd.NA

# --- Convert every variables "numeric" to prevent object errors ---
for col in column_ranges:
    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')



#####################################################################
# ----- MCAR/MAR test ----- # 
#####################################################################

print("\nEksiklikler Arası İlişki Testi (Chi-Square):\n")
for var in column_ranges:
    df_subset[f"{var}_missing"] = df_subset[var].isna()

targets = ["vteurmmb", "lrscale"]
others = [v for v in column_ranges if v not in targets]

for target in targets:
    print(f"\n{target}_missing ilişkili mi?")
    for other in others + [v for v in targets if v != target]:
        contingency = pd.crosstab(df_subset[f"{target}_missing"], df_subset[f"{other}_missing"])
        if contingency.shape == (2, 2):
            try:
                chi2, p, _, _ = chi2_contingency(contingency)
                print(f"- {other}_missing ile ilişki testi: p = {p:.4f}")
            except Exception as e:
                print(f"- {other}_missing ile test yapılamadı: {e}")
        else:
            print(f"- {other}_missing ile test yapılamadı (veri eksik)")

# --- MICE Imputation ---
kernel = mf.ImputationKernel(
    df_subset[list(column_ranges.keys())],
    random_state=42
)
kernel.mice(5)

# --- İmpute edilmiş veriyi al ---
df_imputed = kernel.complete_data(0)

print("\nİmpute edilmiş ilk 5 satır:")
print(df_imputed.head())

# --- İmputasyon sonrası değişkenlerin min-max değerlerini kontrol et ---
print("\nDeğişkenlerin min-max değerleri:")
for col in column_ranges:
    min_val = df_imputed[col].min()
    max_val = df_imputed[col].max()
    has_negative = min_val < 0
    print(f"{col}: Min={min_val:.2f}, Max={max_val:.2f}, Negatif Değer: {has_negative}")



#####################################################################
# ----- FEATURE ENGINEERING ----- #
##################################################################### 
print("\n--- Feature Engineering Başlıyor ---")

# Sonraki adımlar için orjinal veriyi yedekle
df_original = df_imputed.copy()

# Verimizde hangi sütunların mevcut olduğunu kontrol edelim
available_features = df_imputed.columns.tolist()
print("Mevcut özellikler:", available_features)


#####################################################################
# ----- Tutarlı Yön Düzenlemesi - Değişkenler yenilenmeli ------ #
#####################################################################

# Yüksek değerler olumlu olacak şekilde düzenle
direction_reversal = {
    'imdfetn': True,  # 1=allow, 4=don't allow -> yüksek değer negatif
    'imsmetn': True,   # 1=allow, 4=don't allow -> yüksek değer negatif
    'impcntr': True,   # 1=allow, 4=don't allow -> yüksek değer negatif
    'hincfel': True,   # 1=good, 4=bad -> yüksek değer negatif
    'polintr': True    # 1=very interested, 4=not interested -> yüksek değer negatif
}

for col, reverse in direction_reversal.items():
    if col in df_imputed.columns:
        if reverse:
            # Sütunun aralığını belirle
            col_range = column_ranges.get(col, (0, 10))
            # Tersine çevir (örneğin 1-4 aralığında 1->4, 2->3, 3->2, 4->1)
            # Doğrudan sütunu güncelliyoruz, yeni sütun OLUŞTURULMUYOR
            df_imputed[col] = col_range[1] - df_imputed[col] + col_range[0]


#####################################################################
#  trust_institutions  (Ağırlıksız basit ortalama)
#####################################################################

trust_columns = [c for c in ['trstprt', 'trstplt', 'trstprl', 'trstlgl', 'trstplc']
                 if c in df_imputed.columns]

if trust_columns:
    # 1) Basit ortalama → 0-10 ölçeğinde "trust_index"
    df_imputed['institutional_trust'] = df_imputed[trust_columns].mean(axis=1)

    # 2) 0-1 normalizasyon
    min_v, max_v = df_imputed['institutional_trust'].min(), df_imputed['institutional_trust'].max()
    df_imputed['institutional_trust_norm'] = (df_imputed['institutional_trust'] - min_v) / (max_v - min_v)

    # 3) Kontrol – ilk 5 gözlem
    display_cols = trust_columns + ['institutional_trust', 'institutional_trust_norm']
    print(df_imputed[display_cols].head())


#####################################################################
# --- attitudes_towards_immigration ---- #
#####################################################################


# --- 1) Yalnızca 1–4 aralığındaki üç sütunu 0–10 aralığına ölçekle ---
df_imputed['imdfetn_10']  = (df_imputed['imdfetn']  - 1) / (4 - 1) * 10
df_imputed['imsmetn_10'] = (df_imputed['imsmetn'] - 1) / (4 - 1) * 10
df_imputed['impcntr_10'] = (df_imputed['impcntr'] - 1) / (4 - 1) * 10

# --- 2) Diğer 0–10 ölçekli göçmenlik sütunları ---
# (değişkenlerin yönleri değiştirildiği için artık ters eklemiyoruz)
immigration_scaled = [
    'imdfetn_10', 
    'imsmetn_10', 
    'impcntr_10',
    # aşağıdakiler zaten 0–10 aralığında:
    'imwbcnt', 
    'imbgeco', 
    'imueclt'
]

# --- 3) 0–10 ölçekli ortalama ile endeksi üret ---
df_imputed['immigration_openness'] = df_imputed[immigration_scaled].mean(axis=1)

# --- 4) 0–1 aralığında normalize et ---
min_v = df_imputed['immigration_openness'].min()
max_v = df_imputed['immigration_openness'].max()
df_imputed['immigration_openness_norm'] = (
    df_imputed['immigration_openness'] - min_v
) / (max_v - min_v)

# --- 5) Kontrol: ilk 5 satırı yazdır ---
print(df_imputed[immigration_scaled 
                 + ['immigration_openness', 'immigration_openness_norm']
                ].head())



#####################################################################
# ---- national_governance_satisfaction ---- #
#####################################################################

satisfaction_columns = [c for c in ['stfdem', 'stfeco', 'stfedu', 'stfhlth', 'stfgov']
                        if c in df_imputed.columns]

if satisfaction_columns:
    df_imputed['satisfaction_index'] = df_imputed[satisfaction_columns].mean(axis=1)

    min_v, max_v = df_imputed['satisfaction_index'].min(), df_imputed['satisfaction_index'].max()
    df_imputed['satisfaction_index_norm'] = (
        df_imputed['satisfaction_index'] - min_v) / (max_v - min_v)

# 3) Hızlı kontrol
print("\nGöçmenlik sütunları + endeks:")
print(df_imputed[['immigration_openness', 'immigration_openness_norm']].head())

print("\nMemnuniyet sütunları + endeks:")
print(df_imputed[satisfaction_columns + ['satisfaction_index', 'satisfaction_index_norm']].head())



#####################################################################
# ---- eu_sentiment ---- #
#####################################################################

eu_columns = [c for c in ['euftf', 'atcherp', 'trstep'] if c in df_imputed.columns]

if eu_columns:
    # 1) Ağırlıksız ortalama → 0–10 ölçekli sentiment
    df_imputed['eu_sentiment'] = df_imputed[eu_columns].mean(axis=1)

    # 2) 0–1 normalize et
    min_v, max_v = df_imputed['eu_sentiment'].min(), df_imputed['eu_sentiment'].max()
    df_imputed['eu_sentiment_norm'] = (
        df_imputed['eu_sentiment'] - min_v) / (max_v - min_v)

    # 3) Hızlı kontrol
    print(df_imputed[eu_columns + ['eu_sentiment', 'eu_sentiment_norm']].head())

# --- 8. Yaş Tabanlı Segmentasyon ---
if 'agea' in df_imputed.columns:
    df_imputed['age_group'] = pd.cut(
        df_imputed['agea'], 
        bins=[0, 25, 40, 55, 70, 120], 
        labels=['0-25', '26-40', '41-55', '56-70', '71+'],
        right=False
    )
    df_imputed['age_group'] = df_imputed['age_group'].astype('category')



#####################################################################
# ---- distance_from_center =  extremism ---- #
#####################################################################
if 'lrscale' in df_imputed.columns:
    # 1) Merkeze uzaklık
    df_imputed['distance_from_center'] = (df_imputed['lrscale'] - 5).abs()

    # 2) Sola, ortaya, sağa kategorize et (0 da "left"e dahil olsun)
    df_imputed['political_wing'] = pd.cut(
        df_imputed['lrscale'],
        bins=[0, 3, 7, 10],
        labels=['left', 'center', 'right'],
        include_lowest=True   # <-- 0.0 da "left"e sok
    ).astype('category')



#### KONTROL ####

import pandas as pd
pd.set_option('display.max_rows', None)      # Çok sütun varsa hepsini göster

def full_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in df.columns:
        s = df[col]
        # Eğer beklenmedik şekilde DataFrame dönerse ilk alt-sütunu al
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]

        is_num = pd.api.types.is_numeric_dtype(s)
        records.append({
            "feature"       : col,
            "dtype"         : str(s.dtype),
            "missing_count" : s.isna().sum(),
            "missing_%"     : round(s.isna().mean()*100, 2),
            "min"           : s.min() if is_num else None,
            "max"           : s.max() if is_num else None
        })
    return pd.DataFrame(records)

summary = full_column_summary(df_imputed)

print(summary.to_string(index=False))



#####################################################################
# ---- Hedef (vteurmmb) ile kolon korelasyonları ---- #
#####################################################################

target = 'vteurmmb'          # hedef sütun adı

# 1) Sayısal (float, int, bool) sütunları seç
numeric_df = df_imputed.select_dtypes(include=['number', 'bool']).copy()

# 2) Hedef sütun gerçekten sayısal setin içinde mi?
if target not in numeric_df.columns:
    raise ValueError(f"Hedef sütun '{target}' sayısal tipte değil veya bulunamadı.")

# 3) Her sütunun hedefle korelasyonunu hesapla
corr_series = numeric_df.drop(columns=[target]) \
                        .apply(lambda col: col.corr(numeric_df[target]))

# 4) Eksik değerleri at, mutlak değere göre sırala
corr_series = corr_series.dropna()
corr_sorted = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)

# 5) Sonucu yazdır (tam liste + ilk 20)
pd.set_option('display.max_rows', None)   # Tüm satırları görmek isterseniz
print("\n--- Tüm sayısal sütunların hedefle korelasyonları ---")
print(corr_sorted.to_string())

print("\n--- En yüksek 20 korelasyon ---")
print(corr_sorted.head(20))





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 'age' ile başlayan sütunları çıkar
df_filtered = df_imputed.loc[:, ~df_imputed.columns.str.startswith('age')]

# Sadece sayısal (ordinal) sütunlarla çalış (kategorik/string olanları hariç tutmak için)
df_numeric = df_filtered.select_dtypes(include=['number'])

# Spearman korelasyon matrisi
corr_matrix = df_numeric.corr(method='spearman')

# Korelasyon matrisini görselleştir
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Spearman Correlation Matrix")
plt.show()

threshold = 0.6
high_corr = corr_matrix[(abs(corr_matrix) >= threshold) & (abs(corr_matrix) < 1.0)]

# Sadece yüksek korelasyon içeren değerleri yazdır
print("High correlations (|ρ| ≥ 0.6):")
print(high_corr.dropna(how='all').dropna(axis=1, how='all'))
