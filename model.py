"""
model.py - Ball-Outcome Probability Model (XGBoost)
=====================================================
Trains a multi-class classifier to predict the outcome of each delivery.
Classes: dot, single, double, triple, four, six, wicket, extra

Outputs calibrated probabilities per outcome, which feed into the
Monte Carlo simulator for over/innings projections.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

from data_loader import CricketDataLoader
from feature_engine import FeatureEngine


class BallOutcomeModel:
    """
    XGBoost multi-class model for predicting delivery outcomes.
    Outputs probability distribution over 8 outcome types.
    """

    OUTCOME_LABELS = ['dot', 'single', 'double', 'triple',
                      'four', 'six', 'wicket', 'extra']

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series,
              test_size: float = 0.2, random_state: int = 42):
        """
        Train the XGBoost ball-outcome model.

        Uses time-based split: earlier matches for training,
        later matches for testing (prevents data leakage).
        """
        self.feature_names = X.columns.tolist()

        # --- Train/test split ---
        # In production, split by match date. For now, sequential split.
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Training set: {len(X_train):,} deliveries")
        print(f"Test set:     {len(X_test):,} deliveries")
        print(f"Classes:      {sorted(y.unique())}")

        # --- Class weights (handle imbalance) ---
        # Dots and singles dominate; wickets and sixes are rare
        class_counts = y_train.value_counts().sort_index()
        total = len(y_train)
        n_classes = len(class_counts)
        sample_weights = y_train.map(
            {cls: total / (n_classes * count)
             for cls, count in class_counts.items()}
        )

        # --- XGBoost Configuration ---
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='multi:softprob',
            num_class=8,
            eval_metric='mlogloss',
            random_state=random_state,
            n_jobs=-1,
            verbosity=1
        )

        print("\nTraining XGBoost model...")
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        # --- Evaluation ---
        self._evaluate(X_test, y_test)

        # --- Save model ---
        self.save()

        return self.model

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Comprehensive model evaluation."""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # Classification report
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            y_test, y_pred,
            target_names=self.OUTCOME_LABELS,
            digits=3,
            zero_division=0
        ))

        # Log loss (key metric for probability quality)
        ll = log_loss(y_test, y_proba)
        print(f"Log Loss: {ll:.4f}")

        # --- Feature importance ---
        self._plot_feature_importance()

        # --- Calibration check ---
        self._check_calibration(X_test, y_test, y_proba)

        # --- Confusion matrix ---
        self._plot_confusion_matrix(y_test, y_pred)

    def _plot_feature_importance(self):
        """Plot and save top feature importances."""
        importance = self.model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        top_n = min(20, len(feat_imp))
        sns.barplot(data=feat_imp.head(top_n),
                    x='importance', y='feature', color='steelblue')
        plt.title('Top Feature Importances (XGBoost)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'feature_importance.png'),
                    dpi=150)
        plt.close()
        print(f"\nTop 10 features:\n{feat_imp.head(10).to_string(index=False)}")

    def _check_calibration(self, X_test, y_test, y_proba):
        """
        Check if predicted probabilities match actual frequencies.
        Well-calibrated = when model says 20% chance of boundary,
        it actually happens ~20% of the time.
        """
        print("\n--- Calibration Check ---")
        # Check calibration for key outcomes: boundary (4+6) and wicket
        for outcome_idx, label in [(4, 'four'), (5, 'six'), (6, 'wicket')]:
            y_binary = (y_test == outcome_idx).astype(int)
            pred_prob = y_proba[:, outcome_idx]

            if y_binary.sum() < 10:
                print(f"  {label}: too few samples for calibration check")
                continue

            # Predicted vs actual in bins
            bins = np.linspace(0, pred_prob.max(), 6)
            bin_indices = np.digitize(pred_prob, bins)

            for b in range(1, len(bins)):
                mask = bin_indices == b
                if mask.sum() > 0:
                    actual_rate = y_binary[mask].mean()
                    predicted_rate = pred_prob[mask].mean()
                    n = mask.sum()
                    if n >= 20:
                        print(f"  {label} bin {b}: predicted={predicted_rate:.3f}, "
                              f"actual={actual_rate:.3f} (n={n})")

    def _plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.OUTCOME_LABELS,
                    yticklabels=self.OUTCOME_LABELS)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'),
                    dpi=150)
        plt.close()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict outcome probabilities for given feature vectors.

        Returns:
            Array of shape (n_samples, 8) with probabilities for
            [dot, single, double, triple, four, six, wicket, extra]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        return self.model.predict_proba(X)

    def predict_single_ball(self, features: dict) -> dict:
        """
        Predict outcome probabilities for a single delivery.

        Args:
            features: dict with feature names as keys

        Returns:
            Dict mapping outcome labels to probabilities
        """
        X = pd.DataFrame([features])[self.feature_names]
        proba = self.predict_proba(X)[0]
        return dict(zip(self.OUTCOME_LABELS, proba))

    def predict_fast(self, features: dict) -> np.ndarray:
        """
        Fast prediction using numpy array directly (no DataFrame overhead).
        Returns raw probability array of shape (8,).
        Use this in tight simulation loops.
        """
        arr = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        import xgboost as xgb
        dmat = xgb.DMatrix(arr, feature_names=self.feature_names)
        return self.model.get_booster().predict(dmat)[0]

    def save(self, filename: str = "ball_outcome_model.joblib"):
        """Save trained model to disk."""
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, filename)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, path)
        print(f"\nModel saved to {path}")

    def load(self, filename: str = "ball_outcome_model.joblib"):
        """Load trained model from disk."""
        path = os.path.join(self.model_dir, filename)
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        print(f"Model loaded from {path}")


# ----- Full training pipeline -----
if __name__ == "__main__":
    # 1. Load data
    loader = CricketDataLoader(data_dir="data")
    raw = loader.load_all_matches()
    clean = loader.clean_data(raw)
    context = loader.compute_match_context(clean)

    # 2. Build features
    engine = FeatureEngine()
    X, y = engine.build_features(context)

    # 3. Train model
    model = BallOutcomeModel(model_dir="models")
    model.train(X, y)

    # 4. Test single prediction
    print("\n--- Single Ball Prediction Example ---")
    sample_features = X.iloc[1000].to_dict()
    prediction = model.predict_single_ball(sample_features)
    print("Situation features (subset):")
    print(f"  Phase: {'PP' if sample_features.get('is_powerplay') else 'MID' if sample_features.get('is_middle') else 'DEATH'}")
    print(f"  Wickets fallen: {sample_features.get('wickets_fallen', 0):.0f}")
    print(f"  Current RR: {sample_features.get('current_rr', 0):.1f}")
    print(f"  Batter SR: {sample_features.get('bat_sr', 0):.1f}")
    print(f"  Bowler Econ: {sample_features.get('bowl_economy', 0):.1f}")
    print("\nPredicted outcome probabilities:")
    for outcome, prob in sorted(prediction.items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 50)
        print(f"  {outcome:>8s}: {prob:.3f} {bar}")
