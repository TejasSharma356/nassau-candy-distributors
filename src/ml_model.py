"""ML model for predicting whether an order will be delayed."""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODEL_PATH = os.path.join(PROCESSED_DIR, "delay_model.joblib")
ENCODERS_PATH = os.path.join(PROCESSED_DIR, "label_encoders.joblib")


def build_features(df, threshold=7, encoders=None, fit=True):
    """
    Engineer features from the orders dataframe for ML.

    Parameters
    ----------
    df : pd.DataFrame — cleaned orders with factory info merged.
    threshold : int — lead time beyond this = delayed (target=1).
    encoders : dict — pre-fitted LabelEncoders (for inference).
    fit : bool — if True, fit new encoders; if False, use provided ones.

    Returns
    -------
    X, y, encoders, feature_names
    """
    data = df.copy()

    # Target variable
    data['is_delayed'] = (data['lead_time_days'] > threshold).astype(int)

    # Time-based features
    data['order_month'] = pd.to_datetime(data['Order Date']).dt.month
    data['order_dayofweek'] = pd.to_datetime(data['Order Date']).dt.dayofweek
    data['order_quarter'] = pd.to_datetime(data['Order Date']).dt.quarter

    # Categorical columns to encode
    cat_cols = ['Ship Mode', 'Region', 'Division', 'State/Province']
    if 'FACTORY' in data.columns:
        cat_cols.append('FACTORY')

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if col not in data.columns:
            continue
        data[col] = data[col].fillna('Unknown')
        if fit:
            enc = LabelEncoder()
            data[col + '_enc'] = enc.fit_transform(data[col])
            encoders[col] = enc
        else:
            enc = encoders.get(col)
            if enc is not None:
                # Handle unseen labels gracefully
                known = set(enc.classes_)
                data[col + '_enc'] = data[col].apply(
                    lambda v, k=known, e=enc: (
                        e.transform([v])[0] if v in k else -1
                    )
                )
            else:
                data[col + '_enc'] = 0

    # Numeric features
    num_cols = ['Sales', 'Units', 'Cost']
    for col in num_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    feature_names = (
        [c + '_enc' for c in cat_cols if c in data.columns]
        + ['order_month', 'order_dayofweek', 'order_quarter']
        + [c for c in num_cols if c in data.columns]
    )

    x_feats = data[feature_names].values
    y = data['is_delayed'].values

    return x_feats, y, encoders, feature_names


def train_model():
    """Train a Random Forest classifier and save it."""
    # Load data
    orders = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_orders.csv"))
    mapping = pd.read_csv(os.path.join(RAW_DIR, "product_factories.csv"))

    # Merge factory info
    orders = pd.merge(
        orders, mapping, left_on='Division', right_on='DIVISION', how='left'
    )

    # dynamically compute delay threshold
    threshold = orders['lead_time_days'].quantile(0.75)

    print(f"Dataset size: {len(orders)} orders")
    print(f"Delay threshold (75th percentile): {threshold:.1f} days")

    x_feats, y, encoders, feature_names = build_features(
        orders, threshold=threshold, fit=True
    )

    print(f"Delayed: {y.sum()} ({y.mean()*100:.1f}%)  |  "
          f"On-time: {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

    # Train / test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_feats, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    # Evaluate
    y_pred = model.predict(x_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'feature_names': feature_names,
        'threshold': threshold,
        'feature_importances': model.feature_importances_.tolist(),
    }

    print("\n" + classification_report(y_test, y_pred,
          target_names=['On-Time', 'Delayed']))

    # Save
    joblib.dump({'model': model, 'metrics': metrics}, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Encoders saved to {ENCODERS_PATH}")

    return model, metrics, encoders


def load_model():
    """Load the saved model and encoders."""
    bundle = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return bundle['model'], bundle['metrics'], encoders


if __name__ == "__main__":
    train_model()
