"""
03_train_models.py
------------------
Step 3: Train and evaluate classical ML classifiers on the frame feature dataset.

This script follows the standard lecture workflow:
  - Load data with pandas
  - Split into train/test sets
  - Train several scikit-learn classifiers
  - Evaluate each with accuracy, precision, recall, F1, and confusion matrix
  - Save a results summary table

The classification task: predict whether a video frame will be
"hard" (label=1) or "easy" (label=0) for bicubic upscaling to reconstruct well.
This tells us where classical methods struggle — motivating our super-resolution goal.

We also print a separate table of bicubic PSNR/SSIM metrics as our
"classical upscaling baseline."

How to run:
    python 03_train_models.py  (after running 02_build_features.py)
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works without a display)
import matplotlib.pyplot as plt

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier
from sklearn.svm                import SVC
from sklearn.naive_bayes        import GaussianNB
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.metrics            import (accuracy_score, precision_score,
                                         recall_score, f1_score,
                                         confusion_matrix, mean_squared_error,
                                         mean_absolute_error)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
FEATURES_CSV = "output/results/features.csv"
RESULTS_DIR  = "output/results"
IMAGES_DIR   = "output/images"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load the feature dataset
# ─────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(FEATURES_CSV)
print(f"  {len(df)} samples, {len(df.columns)} columns")
print(df.head())

# ─────────────────────────────────────────────
# Separate features (X) and label (y)
# ─────────────────────────────────────────────
# Feature columns — these are the image statistics we computed
feature_cols = ["mean_pixel", "std_pixel", "min_pixel", "max_pixel",
                "contrast", "sharpness", "gradient"]

X = df[feature_cols]
y = df["quality_label"]

print(f"\nClass distribution:\n{y.value_counts()}")

# ─────────────────────────────────────────────
# Train/Test Split (80% train, 20% test)
# Using random_state=42 for reproducibility — same as lecture examples
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# Scale features — important for LR, SVC, KNN
# ─────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# Define the classifiers to test
# These are the same models covered in class lectures
# ─────────────────────────────────────────────
models = {
    "Logistic Regression"   : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"         : DecisionTreeClassifier(random_state=42),
    "Random Forest"         : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVC"                   : SVC(random_state=42),
    "Gaussian Naive Bayes"  : GaussianNB(),
    "K-Nearest Neighbors"   : KNeighborsClassifier(n_neighbors=5),
}

# ─────────────────────────────────────────────
# Train each model, evaluate, store results
# ─────────────────────────────────────────────
results       = []
conf_matrices = {}

print("\n" + "="*65)
print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("="*65)

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate
    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score   (y_test, y_pred, zero_division=0)
    f1   = f1_score       (y_test, y_pred, zero_division=0)
    mse  = mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"{name:<25} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")

    results.append({
        "Model"     : name,
        "Accuracy"  : round(acc,  4),
        "Precision" : round(prec, 4),
        "Recall"    : round(rec,  4),
        "F1 Score"  : round(f1,   4),
        "MSE"       : round(mse,  4),
        "MAE"       : round(mae,  4),
    })
    conf_matrices[name] = cm

print("="*65)

# ─────────────────────────────────────────────
# Save classifier results table
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)

out_path = os.path.join(RESULTS_DIR, "classifier_results.csv")
results_df.to_csv(out_path, index=False)
print(f"\nClassifier results saved to: {out_path}")
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────
# Save bicubic baseline metrics table
# ─────────────────────────────────────────────
bicubic_df = pd.DataFrame([{
    "Method"    : "Bicubic Upscaling",
    "Mean MSE"  : round(df["mse"].mean(),  4),
    "Mean PSNR" : round(df["psnr"].mean(), 2),
    "Mean SSIM" : round(df["ssim"].mean(), 4),
}])
bicubic_path = os.path.join(RESULTS_DIR, "bicubic_metrics.csv")
bicubic_df.to_csv(bicubic_path, index=False)
print(f"\nBicubic baseline metrics saved to: {bicubic_path}")
print(bicubic_df.to_string(index=False))

# ─────────────────────────────────────────────
# Save confusion matrices for all models
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for i, (name, cm) in enumerate(conf_matrices.items()):
    ax = axes[i]
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Easy (0)", "Hard (1)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Easy (0)", "Hard (1)"])
    # Add numbers inside the boxes
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]),
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[row, col] > cm.max() / 2 else "black")

plt.suptitle("Confusion Matrices — All Classifiers", fontsize=13)
plt.tight_layout()
cm_path = os.path.join(IMAGES_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\nConfusion matrices saved to: {cm_path}")

# ─────────────────────────────────────────────
# Save accuracy bar chart
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(results_df["Model"], results_df["Accuracy"], color="steelblue")
ax.set_xlabel("Accuracy")
ax.set_title("Classifier Accuracy Comparison")
ax.set_xlim(0, 1.0)
for i, val in enumerate(results_df["Accuracy"]):
    ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
bar_path = os.path.join(IMAGES_DIR, "accuracy_comparison.png")
plt.savefig(bar_path, dpi=150)
plt.close()
print(f"Accuracy chart saved to: {bar_path}")

print("\nAll done! Check output/results/ and output/images/ for your results.")
