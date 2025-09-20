"""Idea: Fraud Detection
Description:
Fraud detection involves identifying and preventing deceptive activities within financial
transactions or systems. Leveraging advanced analytics and machine learning techniques, fraud
detection systems aim to distinguish between legitimate and fraudulent behavior. Key
components include anomaly detection, pattern recognition, and real-time monitoring."""

# Fraud Detection Pipeline (Kaggle)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. Load dataset
df = pd.read_csv("/kaggle/input/creditcard/creditcard.csv")
print("Data shape:", df.shape)
df.head()

# 2. Feature Engineering
df['Amount_log'] = np.log1p(df['Amount'])
df = df.sort_values('Time').reset_index(drop=True)

df['Amt_rolling_mean_100'] = df['Amount'].rolling(window=100, min_periods=1).mean().fillna(0)
df['Amt_to_roll_mean_ratio'] = df['Amount'] / (df['Amt_rolling_mean_100'] + 1e-6)

v_cols = [c for c in df.columns if c.startswith("V")]
feature_cols = v_cols + ["Amount", "Amount_log", "Amt_rolling_mean_100", "Amt_to_roll_mean_ratio", "Time"]

X = df[feature_cols].fillna(0)
y = df["Class"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Preprocessing (Scaling + PCA)
scaler = StandardScaler()
scaler.fit(X_train[["Amount","Amount_log","Amt_rolling_mean_100","Amt_to_roll_mean_ratio","Time"]])

pca = PCA(n_components=10, random_state=42)
pca.fit(X_train[v_cols])

def preprocess(X_df):
    X_proc = X_df.copy()
    num_cols = ["Amount","Amount_log","Amt_rolling_mean_100","Amt_to_roll_mean_ratio","Time"]
    X_proc[num_cols] = scaler.transform(X_proc[num_cols])
    V_pca = pca.transform(X_proc[v_cols])
    V_pca_df = pd.DataFrame(V_pca, index=X_proc.index, columns=[f"PC{i+1}" for i in range(V_pca.shape[1])])
    keep = ["Amount","Amount_log","Amt_rolling_mean_100","Amt_to_roll_mean_ratio","Time"]
    out = pd.concat([V_pca_df.reset_index(drop=True), X_proc[keep].reset_index(drop=True)], axis=1)
    return out

X_train_prep = preprocess(X_train)
X_test_prep  = preprocess(X_test)

# 5. Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_prep, y_train)
y_pred_lr = lr.predict(X_test_prep)
y_proba_lr = lr.predict_proba(X_test_prep)[:,1]

print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred_lr, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_proba_lr))


# 6. Random Forest
rf = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train_prep, y_train)
y_pred_rf = rf.predict(X_test_prep)
y_proba_rf = rf.predict_proba(X_test_prep)[:,1]

print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_proba_rf))

# 7. Isolation Forest (Anomaly Detection)
X_train_prep_nonfraud = X_train_prep[y_train.values == 0]

iso = IsolationForest(
    n_estimators=100, 
    contamination=y_train.mean(), 
    random_state=42
)
iso.fit(X_train_prep_nonfraud)

iso_pred = iso.predict(X_test_prep)
iso_labels = (iso_pred == -1).astype(int)

print("\nIsolation Forest Results:")
print(classification_report(y_test, iso_labels, digits=4))

# 8. Confusion Matrix (for RF)
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. Real-time prediction simulation
def predict_transaction(model, tx):
    """
    tx: dict with keys ['V', 'Amount', 'Time']
    V must be a list of 28 PCA features.
    """
    v_arr = np.array(tx['V']).reshape(1,-1)
    v_pca = pca.transform(v_arr)
    amt_log = np.log1p(tx['Amount'])
    amt_roll = tx['Amount']
    amt_ratio = tx['Amount'] / (amt_roll + 1e-6)
    num = np.array([[tx['Amount'], amt_log, amt_roll, amt_ratio, tx['Time']]])
    num_scaled = scaler.transform(num)
    X_final = np.hstack([v_pca, num_scaled])
    prob = model.predict_proba(X_final)[0,1]
    return prob

# Example
sample_tx = {
    "V": list(X_test[v_cols].iloc[0]), 
    "Amount": float(X_test.iloc[0]["Amount"]),
    "Time": float(X_test.iloc[0]["Time"])
}
print("\nSample transaction fraud probability (RF):", predict_transaction(rf, sample_tx))
