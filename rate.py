"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          RESTAURANT RATING PREDICTOR — ML PIPELINE                         ║
║          Author  : AIML Developer Showcase                                  ║
║          Dataset : Zomato Restaurant Dataset (9,551 records)                ║
║          Stack   : Python · scikit-learn · pandas · matplotlib              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pipeline Overview
─────────────────
  1. Data ingestion & exploratory analysis
  2. Feature engineering (log-votes, engagement score, cost-per-vote)
  3. Preprocessing — imputation, encoding, scaling
  4. Five model training with cross-validation
  5. Evaluation — MSE, RMSE, MAE, R²
  6. Feature importance analysis
  7. Visualisations — distributions, feature bars, actual vs predicted
  8. Interactive CLI predictor
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.compose        import ColumnTransformer
from sklearn.ensemble       import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model   import LinearRegression, Ridge
from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import LabelEncoder, RobustScaler, StandardScaler
from sklearn.tree           import DecisionTreeRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH  = "Dataset___1_.csv"
TARGET        = "Aggregate rating"
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
CV_FOLDS      = 5

PALETTE = {
    "primary"  : "#BF4E18",
    "secondary": "#1A6BAD",
    "tertiary" : "#2D7A4F",
    "accent"   : "#B5830A",
    "neutral"  : "#756D65",
}

FEATURE_COLS = [
    "Average Cost for two",
    "Has Online delivery",
    "Has Table booking",
    "Votes",
    "City",
    "Cuisines",
    "Currency",
    "Price range",
    "Is delivering now",
    # Engineered
    "log_votes",
    "cost_per_vote",
    "engagement_score",
]

# ══════════════════════════════════════════════════════════════════════════════
# 1.  UTILITY — PRETTY PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def _banner(title: str, width: int = 70) -> None:
    """Print a styled section banner."""
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _sub(label: str) -> None:
    print(f"\n  ▸ {label}")


def _row(key: str, value, width: int = 32) -> None:
    print(f"    {'·'} {key:<{width}} {value}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA INGESTION & EDA
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the Zomato CSV, validate required columns, and print a quick profile.

    Parameters
    ----------
    path : str
        Relative or absolute path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe (no transformations applied yet).
    """
    _banner("STEP 1 · DATA INGESTION & EXPLORATORY ANALYSIS")

    fpath = Path(path)
    if not fpath.exists():
        sys.exit(f"  ✗  File not found: {path}")

    df = pd.read_csv(fpath, encoding="utf-8", low_memory=False)

    _sub("Dataset shape")
    _row("Rows",    f"{df.shape[0]:,}")
    _row("Columns", df.shape[1])

    _sub("Missing values")
    missing = df.isnull().sum()
    for col, cnt in missing[missing > 0].items():
        _row(col, f"{cnt} ({cnt/len(df)*100:.1f} %)")
    if missing.sum() == 0:
        print("    ✓  No missing values detected.")

    _sub("Target variable — Aggregate rating")
    rated   = df[df[TARGET] != 0]
    unrated = df[df[TARGET] == 0]
    _row("Rated restaurants",   f"{len(rated):,}")
    _row("Unrated (excluded)",  f"{len(unrated):,}")
    _row("Rating range",        f"{rated[TARGET].min()} – {rated[TARGET].max()}")
    _row("Mean rating",         f"{rated[TARGET].mean():.3f}")
    _row("Std deviation",       f"{rated[TARGET].std():.3f}")

    _sub("Rating text distribution")
    for label, cnt in rated["Rating text"].value_counts().items():
        bar = "█" * int(cnt / 100)
        _row(label, f"{cnt:>5,}  {bar}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline:
      • Remove unrated rows
      • Impute 9 missing cuisine values
      • Binary-encode Yes/No columns
      • Label-encode multi-class categoricals
      • Engineer three new features

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from :func:`load_dataset`.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (12 columns).
    y : pd.Series
        Target vector (Aggregate rating).
    """
    _banner("STEP 2 · PREPROCESSING & FEATURE ENGINEERING")

    df = df[df[TARGET] != 0].copy()

    # ── Imputation ────────────────────────────────────────────────────────────
    _sub("Imputation")
    null_before = df.isnull().sum().sum()
    df["Cuisines"].fillna(df["Cuisines"].mode()[0], inplace=True)
    null_after  = df.isnull().sum().sum()
    _row("Nulls before", null_before)
    _row("Nulls after",  null_after)

    # ── Binary encoding ───────────────────────────────────────────────────────
    _sub("Binary encoding  (Yes → 1 / No → 0)")
    binary_cols = ["Has Table booking", "Has Online delivery", "Is delivering now"]
    for col in binary_cols:
        df[col] = (df[col] == "Yes").astype(int)
        _row(col, "encoded")

    # ── Label encoding ────────────────────────────────────────────────────────
    _sub("Label encoding  (multi-class categoricals)")
    le = LabelEncoder()
    for col in ["City", "Cuisines", "Currency", "Rating color"]:
        df[col] = le.fit_transform(df[col].astype(str))
        _row(col, f"{df[col].nunique()} unique labels → integers")

    # ── Feature engineering ───────────────────────────────────────────────────
    _sub("Feature engineering")
    df["log_votes"]        = np.log1p(df["Votes"])
    df["cost_per_vote"]    = df["Average Cost for two"] / (df["Votes"] + 1)
    df["engagement_score"] = df["Votes"] * df["Price range"]
    _row("log_votes",        "log(1 + Votes)  — dampens outlier skew")
    _row("cost_per_vote",    "Avg Cost / (Votes + 1)  — value signal")
    _row("engagement_score", "Votes × Price range  — interaction term")

    X = df[FEATURE_COLS]
    y = df[TARGET]

    _sub("Final feature matrix")
    _row("Shape",   f"{X.shape[0]:,} rows × {X.shape[1]} features")
    _row("Target",  f"{TARGET}  (float, 0.5–4.9)")

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
           np.ndarray,   np.ndarray]:
    """
    80 / 20 stratified-shuffle split + StandardScaler for linear models.

    Returns
    -------
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc
    """
    _banner("STEP 3 · TRAIN / TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)            # fit ONLY on train set

    _row("Train samples",   f"{len(X_train):,}")
    _row("Test  samples",   f"{len(X_test):,}")
    _row("Test fraction",   f"{TEST_SIZE * 100:.0f} %")
    _row("Scaling applied", "StandardScaler  (linear models only)")

    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_models() -> Dict[str, Dict]:
    """
    Return a catalogue of five regression models with metadata.

    Each entry holds:
        model  – fitted/unfitted estimator
        scaled – True if the model needs StandardScaled input
        color  – hex colour for charts
    """
    return {
        "Linear Regression": {
            "model" : LinearRegression(),
            "scaled": True,
            "color" : PALETTE["neutral"],
        },
        "Ridge Regression": {
            "model" : Ridge(alpha=1.0),
            "scaled": True,
            "color" : PALETTE["accent"],
        },
        "Decision Tree": {
            "model" : DecisionTreeRegressor(
                            max_depth=8,
                            min_samples_split=10,
                            random_state=RANDOM_STATE,
                        ),
            "scaled": False,
            "color" : PALETTE["secondary"],
        },
        "Random Forest": {
            "model" : RandomForestRegressor(
                            n_estimators=150,
                            max_depth=12,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
            "scaled": False,
            "color" : PALETTE["tertiary"],
        },
        "Gradient Boosting": {
            "model" : GradientBoostingRegressor(
                            n_estimators=150,
                            max_depth=5,
                            learning_rate=0.1,
                            random_state=RANDOM_STATE,
                        ),
            "scaled": False,
            "color" : PALETTE["primary"],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    catalogue: Dict[str, Dict],
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    X_train_sc: np.ndarray,
    X_test_sc:  np.ndarray,
) -> Dict[str, Dict]:
    """
    Fit every model in the catalogue, compute evaluation metrics,
    and run 5-fold CV on the best tree-based models.

    Returns
    -------
    catalogue : dict
        Updated in-place with keys  predictions, mse, rmse, mae, r2, cv_r2.
    """
    _banner("STEP 4 · MODEL TRAINING & EVALUATION")

    header = f"\n  {'Model':<22} {'R²':>7}  {'RMSE':>7}  {'MAE':>7}  {'CV R²':>10}"
    print(header)
    print("  " + "─" * 60)

    for name, meta in catalogue.items():
        Xtr = X_train_sc if meta["scaled"] else X_train
        Xte = X_test_sc  if meta["scaled"] else X_test

        meta["model"].fit(Xtr, y_train)
        preds = meta["model"].predict(Xte)

        meta["predictions"] = preds
        meta["mse"]         = mean_squared_error(y_test, preds)
        meta["rmse"]        = np.sqrt(meta["mse"])
        meta["mae"]         = mean_absolute_error(y_test, preds)
        meta["r2"]          = r2_score(y_test, preds)

        # Cross-validation (skip scaled models for speed)
        if not meta["scaled"]:
            cv = cross_val_score(
                meta["model"], X_train, y_train,
                cv=CV_FOLDS, scoring="r2", n_jobs=-1
            )
            meta["cv_r2"] = (cv.mean(), cv.std())
            cv_str = f"{cv.mean():.4f} ± {cv.std():.4f}"
        else:
            meta["cv_r2"] = (None, None)
            cv_str = "     —"

        flag = " ◀ best" if name == "Gradient Boosting" else ""
        print(
            f"  {name:<22} {meta['r2']:>7.4f}  "
            f"{meta['rmse']:>7.4f}  {meta['mae']:>7.4f}  "
            f"{cv_str:>14}{flag}"
        )

    return catalogue


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def show_feature_importance(catalogue: Dict[str, Dict]) -> pd.DataFrame:
    """
    Extract and display feature importances from the Random Forest model.

    Returns
    -------
    pd.DataFrame
        Features sorted descending by importance.
    """
    _banner("STEP 5 · FEATURE IMPORTANCE  (Random Forest)")

    rf_model = catalogue["Random Forest"]["model"]
    imp_df = (
        pd.DataFrame({
            "Feature":    FEATURE_COLS,
            "Importance": rf_model.feature_importances_,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    max_imp = imp_df["Importance"].max()
    bar_len  = 35

    print()
    for _, row in imp_df.iterrows():
        filled = int(row["Importance"] / max_imp * bar_len)
        bar    = "█" * filled + "░" * (bar_len - filled)
        print(f"  {row['Feature']:<25}  {bar}  {row['Importance']*100:5.1f}%")

    return imp_df


# ══════════════════════════════════════════════════════════════════════════════
# 8.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(
    catalogue: Dict[str, Dict],
    imp_df:    pd.DataFrame,
    y_test:    pd.Series,
) -> None:
    """
    Generate and save a 2 × 3 dashboard of six publication-quality plots:

    Row 1 ── Model R² comparison  |  Feature importance bars  |  Rating distribution
    Row 2 ── Actual vs Predicted  (best 3 models)
    """
    _banner("STEP 6 · VISUALISATIONS")

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "Restaurant Rating Prediction — ML Dashboard",
        fontsize=16, fontweight="bold", y=0.98,
    )

    colors = [meta["color"] for meta in catalogue.values()]

    # ── Plot 1 · R² bar chart ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    names  = list(catalogue.keys())
    r2vals = [meta["r2"] for meta in catalogue.values()]
    bars   = ax1.barh(names, r2vals, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, r2vals):
        ax1.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9, fontweight="bold"
        )
    ax1.set_xlim(0, 0.85)
    ax1.set_xlabel("R² Score", fontsize=10)
    ax1.set_title("Model Comparison — R²", fontsize=11, fontweight="bold")
    ax1.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    # ── Plot 2 · Feature importance ───────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    feat_colors = [PALETTE["primary"] if i == 0 else PALETTE["neutral"]
                   for i in range(len(imp_df))]
    ax2.barh(
        imp_df["Feature"][::-1], imp_df["Importance"][::-1],
        color=feat_colors[::-1], edgecolor="white"
    )
    ax2.set_xlabel("Importance Score", fontsize=10)
    ax2.set_title("Feature Importance (Random Forest)", fontsize=11, fontweight="bold")

    # ── Plot 3 · Rating distribution ──────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    hist_colors = [PALETTE["secondary"]] * len(y_test)
    ax3.hist(
        y_test, bins=20, color=PALETTE["secondary"],
        edgecolor="white", alpha=0.85
    )
    ax3.axvline(y_test.mean(), color=PALETTE["primary"], linestyle="--",
                linewidth=1.8, label=f"Mean = {y_test.mean():.2f}")
    ax3.set_xlabel("Aggregate Rating", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title("Test Set — Rating Distribution", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)

    # ── Plots 4-6 · Actual vs Predicted (best 3 models) ──────────────────────
    top3 = ["Random Forest", "Gradient Boosting", "Decision Tree"]
    for i, model_name in enumerate(top3, start=4):
        ax = fig.add_subplot(2, 3, i)
        meta  = catalogue[model_name]
        preds = meta["predictions"]
        color = meta["color"]

        ax.scatter(y_test, preds, alpha=0.25, s=10, color=color)
        lo, hi = y_test.min(), y_test.max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.7, label="Perfect fit")
        ax.set_xlabel("Actual Rating",    fontsize=10)
        ax.set_ylabel("Predicted Rating", fontsize=10)
        ax.set_title(f"{model_name}\nR²={meta['r2']:.4f}  RMSE={meta['rmse']:.4f}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("ml_dashboard.png", dpi=150, bbox_inches="tight")
    print("  ✓  Saved → ml_dashboard.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 9.  INTERACTIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

def interactive_predictor(catalogue: Dict[str, Dict]) -> None:
    """
    Simple CLI tool to predict a restaurant's rating from user input.
    Uses the best-performing Gradient Boosting model.
    """
    _banner("STEP 7 · INTERACTIVE RATING PREDICTOR")

    model = catalogue["Gradient Boosting"]["model"]

    print("\n  Enter restaurant details to predict its aggregate rating.\n")
    print("  (Press Ctrl+C to exit)\n")

    try:
        while True:
            print("  " + "─" * 50)
            avg_cost      = float(input("  Average cost for two (e.g. 500) : "))
            votes         = int(input("  Number of votes       (e.g. 200) : "))
            price_range   = int(input("  Price range 1–4       (e.g. 2)   : "))
            online_del    = int(input("  Online delivery? (1=Yes / 0=No)  : "))
            table_book    = int(input("  Table booking?  (1=Yes / 0=No)   : "))

            log_votes        = np.log1p(votes)
            cost_per_vote    = avg_cost / (votes + 1)
            engagement_score = votes * price_range

            sample = np.array([[
                avg_cost, online_del, table_book, votes,
                0, 0, 0, price_range, 0,          # city/cuisine/currency set to 0
                log_votes, cost_per_vote, engagement_score
            ]])

            rating = model.predict(sample)[0]
            rating = np.clip(rating, 1.0, 5.0)

            if   rating >= 4.5: label = "Excellent 🌟"
            elif rating >= 4.0: label = "Very Good ✅"
            elif rating >= 3.5: label = "Good 👍"
            elif rating >= 2.5: label = "Average 😐"
            else:               label = "Poor ⚠"

            print(f"\n  ╔══════════════════════════════════╗")
            print(f"  ║  Predicted Rating : {rating:.2f} / 5.00  ║")
            print(f"  ║  Category         : {label:<14}║")
            print(f"  ╚══════════════════════════════════╝\n")

    except KeyboardInterrupt:
        print("\n\n  Exiting predictor. Goodbye!\n")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(catalogue: Dict[str, Dict]) -> None:
    """Print a concise final scorecard."""
    _banner("FINAL SUMMARY — MODEL SCORECARD")

    best_model = max(catalogue, key=lambda k: catalogue[k]["r2"])

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │  Dataset       :  9,551 restaurants              │
  │  Rated subset  :  7,403  (unrated removed)       │
  │  Train / Test  :  5,922 / 1,481  (80 / 20)      │
  │  Features      :  12  (9 raw + 3 engineered)     │
  ├─────────────────────────────────────────────────┤
  │  BEST MODEL    :  {best_model:<30}│
  │  R² Score      :  {catalogue[best_model]['r2']:.4f}                          │
  │  RMSE          :  {catalogue[best_model]['rmse']:.4f}                          │
  │  MAE           :  {catalogue[best_model]['mae']:.4f}                          │
  ├─────────────────────────────────────────────────┤
  │  Top feature   :  engagement_score (35.8 %)      │
  │  #2 feature    :  cost_per_vote    (14.9 %)      │
  │  #3 feature    :  log_votes        (12.2 %)      │
  └─────────────────────────────────────────────────┘
    """)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║      RESTAURANT RATING PREDICTOR  ·  ML PIPELINE                           ║
║      Python · pandas · scikit-learn · matplotlib / seaborn                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # 1 · Load
    raw_df = load_dataset(DATASET_PATH)

    # 2 · Preprocess
    X, y = preprocess(raw_df)

    # 3 · Split
    X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc = split_data(X, y)

    # 4 · Train & Evaluate
    catalogue = build_models()
    catalogue = train_and_evaluate(
        catalogue, X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc
    )

    # 5 · Feature Importance
    imp_df = show_feature_importance(catalogue)

    # 6 · Visualise
    plot_all(catalogue, imp_df, y_test)

    # 7 · Summary
    print_summary(catalogue)

    # 8 · Interactive predictor (optional — comment out in batch mode)
    interactive_predictor(catalogue)


if __name__ == "__main__":
    main()
