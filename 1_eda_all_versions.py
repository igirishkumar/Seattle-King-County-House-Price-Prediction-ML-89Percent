# 1_eda_all_versions.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import os


# === 1. Load & Split ===
df = pd.read_csv("king_ country_ houses_aa.csv")
df = df.drop(columns=['id'], errors='ignore')

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Preprocess (8 versions) ===
def preprocess_version(df_in, version, fit=False):
    df = df_in.copy()

    # Date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year']  = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['dow']   = df['date'].dt.dayofweek
        df = df.drop('date', axis=1)

    numeric = ['sqft_living','sqft_above','sqft_basement','lat','long',
               'sqft_living15','sqft_lot15','yr_built','yr_renovated',
               'bathrooms','bedrooms','floors']
    if 'year' in df.columns: numeric += ['year','month','dow']
    ordinal = ['grade','view']
    binary  = ['waterfront']
    keep = numeric + ordinal + binary
    df = df[keep]

    # Outlier removal
    use_out = version.startswith('Out') or version.startswith('Clean')
    if use_out:
        if fit:
            global iqr_bounds
            iqr_bounds = {}
            for c in numeric:
                Q1, Q3 = df[c].quantile([0.25,0.75])
                IQR = Q3 - Q1
                lo, hi = Q1-1.5*IQR, Q3+1.5*IQR
                iqr_bounds[c] = (lo, hi)
                df = df[(df[c] >= lo) & (df[c] <= hi)]
        else:
            for c in numeric:
                lo, hi = iqr_bounds[c]
                df = df[(df[c] >= lo) & (df[c] <= hi)]

    # Scaling
    use_scale = 'Norm' in version or 'Clean' in version
    if use_scale:
        if fit:
            global means, stds
            means = df[numeric].mean()
            stds  = df[numeric].std()
            stds.replace(0, 1, inplace=True)   # prevent div-0
        df[numeric] = (df[numeric] - means) / stds

    # Binary
    df['waterfront_1'] = df['waterfront']
    df = df.drop('waterfront', axis=1)

    # Feature Selection
    use_select = version.endswith('Selected')
    if use_select:
        if fit:
            global selected_cols
            X_num = df[numeric].copy()

            # 1. Drop zero-variance columns
            var = X_num.var()
            X_num = X_num.loc[:, var > 1e-8]

            # 2. If still too few → fallback to correlation
            if X_num.shape[1] < 12:
                corr = X_num.corrwith(y_train.loc[X_num.index]).abs().sort_values(ascending=False)
                cols = corr.index[:12].tolist()
            else:
                selector = SelectKBest(f_classif, k=min(12, X_num.shape[1]))
                selector.fit(X_num, y_train.loc[X_num.index])
                cols = X_num.columns[selector.get_support()].tolist()

            selected_cols = cols
            df = df[selected_cols + ordinal]

        else:
            df = df[selected_cols + ordinal]

    return df


# === 3. Save All 8 Versions + Plots ===
versions = ["Raw_All","Raw_Selected","Out_All","Out_Selected",
            "Norm_All","Norm_Selected","Clean_All","Clean_Selected"]

os.makedirs("versions", exist_ok=True)

for v in versions:
    print(f"\n=== {v} ===")
    X_tr = preprocess_version(X_train, v, fit=True)
    X_tr['price'] = y_train.loc[X_tr.index]
    X_tr.to_csv(f"versions/{v}_train.csv", index=False)

    X_te = preprocess_version(X_test, v, fit=False)
    X_te['price'] = y_test.loc[X_te.index]
    X_te.to_csv(f"versions/{v}_test.csv", index=False)

    print(f"Saved {v} → {X_tr.shape}")

    # Quick plot
    plt.figure(figsize=(9,5))
    sns.scatterplot(data=X_tr, x='sqft_living', y='price', alpha=0.5, label='train')
    sns.scatterplot(data=X_te, x='sqft_living', y='price', alpha=0.5, color='red', label='test')
    plt.title(f"{v}")
    plt.legend()
    plt.savefig(f"versions/plot_{v}.png")
    plt.close()

# # === 5. Track selected features per version ===
# selected_features = {}

# for v in versions:
#     print(f"\n=== {v} ===")
    
#     # --- Fit on train ---
#     X_tr = preprocess_version(X_train, v, fit=True)
#     X_tr['price'] = y_train.loc[X_tr.index]
#     X_tr.to_csv(f"versions/{v}_train.csv", index=False)

#     # --- Save selected features (only for Selected versions) ---
#     if v.endswith('Selected') and 'selected_cols' in globals():
#         selected_features[v] = selected_cols
#     else:
#         selected_features[v] = X_tr.drop('price', axis=1).columns.tolist()

#     # --- Test ---
#     X_te = preprocess_version(X_test, v, fit=False)
#     X_te['price'] = y_test.loc[X_te.index]
#     X_te.to_csv(f"versions/{v}_test.csv", index=False)

#     print(f"Saved {v} → {X_tr.shape}")

print("\nALL 8 VERSIONS READY! Run 2_modeling_all_versions.ipynb next.")


# --- 4. Fit on train ---
X_train_clean = preprocess_version(X_train, v, fit=True)
X_test_clean  = preprocess_version(X_test,  v, fit=False)

X_train_clean['price'] = y_train.loc[X_train_clean.index]
X_test_clean['price']  = y_test.loc[X_test_clean.index]

print(f"Train: {X_train_clean.shape}, Test: {X_test_clean.shape}")

# Price vs sqft_living
plt.figure(figsize=(10,6))
sns.scatterplot(data=X_train_clean, x='sqft_living', y='price', alpha=0.6)
plt.title("Price vs Sqft Living")
plt.show()

# Price by grade
plt.figure(figsize=(10,6))
sns.boxplot(data=X_train_clean, x='grade', y='price')
plt.title("Price by Grade")
plt.show()

# # Waterfront effect
# plt.figure(figsize=(8,5))
# sns.boxplot(data=X_train_clean, x='waterfront', y='price')
# plt.title("Price: Waterfront vs No")
# plt.xlabel("Waterfront")
# plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(X_train_clean.corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.show()


# ==============================================================
#  RUN THIS AFTER you have executed 1_eda_all_versions_FIXED.py
#  (or after the `versions/` folder exists with all CSVs)
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ------------------------------------------------------------------
# 1. Load the 8 saved train CSVs
# ------------------------------------------------------------------
versions = [
    "Raw_All", "Raw_Selected", "Out_All", "Out_Selected",
    "Norm_All", "Norm_Selected", "Clean_All", "Clean_Selected"
]

eda_dfs = {}
for v in versions:
    path = f"versions/{v}_train.csv"
    if os.path.exists(path):
        eda_dfs[v] = pd.read_csv(path)
    else:
        print(f"Warning: {path} not found – skipping {v}")

# ------------------------------------------------------------------
# 2. Re-construct `selected_features` from the column names
# ------------------------------------------------------------------
selected_features = {}
for v, df in eda_dfs.items():
    # All columns except price
    feats = df.drop(columns='price', errors='ignore').columns.tolist()
    selected_features[v] = feats

# ------------------------------------------------------------------
# 3. Paste the plot_eda function (exactly as you posted)
# ------------------------------------------------------------------
def plot_eda(df, name, selected_features_dict):
    print(f"\n{'='*20} {name} {'='*20}")
    print(f"Shape: {df.shape}")
    print(f"Selected Features: {selected_features_dict.get(name, 'All')}")

    target = 'price'
    if target not in df.columns:
        print(f"Warning: '{target}' not in {name} — skipping price plots")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.drop(target, errors='ignore')

    # 1. Histograms
    n_cols = len(numeric_cols)
    if n_cols > 0:
        df[numeric_cols].hist(figsize=(15, 10), bins=30,
                              color='skyblue', edgecolor='black')
        plt.suptitle(f"Feature Distributions - {name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # 2. Boxplots
    if n_cols > 0:
        plt.figure(figsize=(min(15, n_cols * 1.5), 6))
        sns.boxplot(data=df[numeric_cols], palette="Set2")
        plt.title(f"Boxplots - {name}", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # 3. Correlation with price
    if target in df.columns and n_cols > 0:
        corr = df[numeric_cols].corrwith(df[target]).sort_values(ascending=False)
        plt.figure(figsize=(8, max(4, len(corr) * 0.3)))
        corr.plot(kind='barh', color='coral',
                  title=f"Correlation with Price - {name}")
        plt.xlabel("Correlation")
        plt.tight_layout()
        plt.show()

        # 4. Price vs Top 3 Features
        top3 = corr.index[:3].tolist()
        if top3:
            n_plot = min(3, len(top3))
            fig, axes = plt.subplots(1, n_plot, figsize=(6 * n_plot, 5))
            if n_plot == 1:
                axes = [axes]
            for i, col in enumerate(top3):
                sns.scatterplot(data=df, x=col, y=target,
                                alpha=0.6, ax=axes[i], color='teal')
                axes[i].set_title(f"{col} vs Price", fontsize=12)
            plt.tight_layout()
            plt.show()

# ------------------------------------------------------------------
# 4. Run EDA for every version
# ------------------------------------------------------------------
for name, df in eda_dfs.items():
    plot_eda(df, name, selected_features)

    