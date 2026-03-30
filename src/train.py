from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

def train_baseline_rf(df: pd.DataFrame, random_state: int = 42) -> tuple:
    """Trains the baseline Random Forest model."""
    X = df[['recency', 'frequency', 'monetary']]
    y = df['target']
    
    # Stratified split to maintain class balance
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=random_state)) #
    ])
    
    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict_proba(X_val)[:, 1]
    
    metrics = {
        'auc': roc_auc_score(y_val, val_preds)
    }
    
    return pipeline, metrics

# In a real project, retrain_challenger can wrap train_baseline_rf but take cumulative data arrays.
def train_challenger(df: pd.DataFrame, random_state: int = 42) -> tuple:
    """Wraps baseline training for challenger models."""
    return train_baseline_rf(df, random_state)