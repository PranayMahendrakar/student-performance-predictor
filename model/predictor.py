"""
Student Performance Predictor - Inference Engine
Loads trained models and provides prediction + improvement suggestions.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from typing import Dict, Any, Tuple, List


FEATURE_COLUMNS = [
    "attendance_rate",
    "assignment_avg",
    "midterm_score",
    "study_hours_per_week",
    "participation_score",
    "previous_gpa",
    "assignment_completion",
    "quiz_avg",
]

FEATURE_LABELS = {
    "attendance_rate": "Attendance Rate (%)",
    "assignment_avg": "Assignment Average (%)",
    "midterm_score": "Midterm Score (%)",
    "study_hours_per_week": "Study Hours / Week",
    "participation_score": "Participation Score (0-10)",
    "previous_gpa": "Previous GPA (0.0-4.0)",
    "assignment_completion": "Assignment Completion (%)",
    "quiz_avg": "Quiz Average (%)",
}

PERFORMANCE_COLORS = {
    "Pass": "#28a745",
    "At Risk": "#fd7e14",
    "Fail": "#dc3545",
}

PERFORMANCE_ICONS = {
    "Pass": "✅",
    "At Risk": "⚠️",
    "Fail": "❌",
}

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


class StudentPredictor:
    """Wraps RF and GBM models for inference and suggestion generation."""

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.rf = None
        self.gbm = None
        self.scaler = None
        self.le = None
        self.metadata = {}
        self._load()

    def _load(self):
        """Load model artifacts from disk."""
        try:
            self.rf = joblib.load(os.path.join(self.artifacts_dir, "random_forest.pkl"))
            self.gbm = joblib.load(os.path.join(self.artifacts_dir, "gradient_boosting.pkl"))
            self.scaler = joblib.load(os.path.join(self.artifacts_dir, "scaler.pkl"))
            self.le = joblib.load(os.path.join(self.artifacts_dir, "label_encoder.pkl"))
            meta_path = os.path.join(self.artifacts_dir, "model_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self.metadata = json.load(f)
        except FileNotFoundError:
            # Models not yet trained - use synthetic prediction
            pass

    @property
    def is_loaded(self) -> bool:
        return all([self.rf, self.gbm, self.scaler, self.le])

    def predict(
        self,
        input_data: Dict[str, float],
        model_choice: str = "random_forest",
    ) -> Dict[str, Any]:
        """
        Run inference on a single student record.
        Returns prediction, probabilities, and improvement suggestions.
        """
        df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

        if not self.is_loaded:
            return self._fallback_predict(input_data)

        X_scaled = self.scaler.transform(df)
        model = self.rf if model_choice == "random_forest" else self.gbm

        pred_idx = model.predict(X_scaled)[0]
        pred_label = self.le.inverse_transform([pred_idx])[0]
        proba = model.predict_proba(X_scaled)[0]
        class_probs = {
            self.le.classes_[i]: float(p) for i, p in enumerate(proba)
        }

        # Feature importances for this model
        importances = dict(
            zip(FEATURE_COLUMNS, model.feature_importances_)
        )

        suggestions = self._generate_suggestions(input_data, pred_label, importances)
        risk_factors = self._identify_risk_factors(input_data)

        return {
            "prediction": pred_label,
            "confidence": float(max(proba)),
            "probabilities": class_probs,
            "suggestions": suggestions,
            "risk_factors": risk_factors,
            "feature_importances": importances,
            "color": PERFORMANCE_COLORS.get(pred_label, "#6c757d"),
            "icon": PERFORMANCE_ICONS.get(pred_label, "❓"),
        }

    def predict_both(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """Run both RF and GBM and return combined results."""
        rf_result = self.predict(input_data, "random_forest")
        gbm_result = self.predict(input_data, "gradient_boosting")
        # Ensemble average
        classes = list(rf_result["probabilities"].keys())
        ensemble_proba = {
            c: (rf_result["probabilities"][c] + gbm_result["probabilities"][c]) / 2
            for c in classes
        }
        ensemble_pred = max(ensemble_proba, key=ensemble_proba.get)
        return {
            "random_forest": rf_result,
            "gradient_boosting": gbm_result,
            "ensemble": {
                "prediction": ensemble_pred,
                "confidence": float(max(ensemble_proba.values())),
                "probabilities": ensemble_proba,
                "color": PERFORMANCE_COLORS.get(ensemble_pred, "#6c757d"),
                "icon": PERFORMANCE_ICONS.get(ensemble_pred, "❓"),
            },
        }

    # ── Suggestion engine ──────────────────────────────────────────────────────
    def _generate_suggestions(
        self,
        data: Dict[str, float],
        prediction: str,
        importances: Dict[str, float],
    ) -> List[Dict[str, str]]:
        suggestions = []
        sorted_features = sorted(importances, key=importances.get, reverse=True)

        # Threshold-based rules (ordered by importance)
        rules = {
            "attendance_rate": (75, "Boost attendance to at least 75%. Each missed class compounds learning gaps."),
            "study_hours_per_week": (10, "Aim for 10+ study hours per week. Use focused Pomodoro sessions."),
            "assignment_completion": (85, "Submit at least 85% of assignments - they directly drive your grade."),
            "assignment_avg": (60, "Revisit graded assignments and seek feedback to lift your average above 60%."),
            "midterm_score": (60, "Schedule a meeting with your instructor to review midterm mistakes."),
            "quiz_avg": (60, "Practice past quizzes weekly - short quizzes account for long-term retention."),
            "participation_score": (5, "Participate more in class discussions to reinforce understanding."),
            "previous_gpa": (2.0, "Speak to your academic advisor about strategies to improve your GPA trajectory."),
        }

        for feat in sorted_features:
            if feat in rules:
                threshold, msg = rules[feat]
                if data.get(feat, threshold + 1) < threshold:
                    icon = "📚" if "study" in feat else "📋" if "assign" in feat else "🎯"
                    suggestions.append({
                        "feature": FEATURE_LABELS.get(feat, feat),
                        "current": round(data.get(feat, 0), 1),
                        "target": threshold,
                        "message": msg,
                        "icon": icon,
                        "priority": "High" if importances[feat] > 0.15 else "Medium",
                    })

        if prediction == "Pass" and not suggestions:
            suggestions.append({
                "feature": "Overall",
                "current": None,
                "target": None,
                "message": "Excellent performance! Maintain your current habits and consider peer tutoring.",
                "icon": "🌟",
                "priority": "Low",
            })

        return suggestions[:6]  # Top 6 suggestions

    def _identify_risk_factors(self, data: Dict[str, float]) -> List[str]:
        factors = []
        if data.get("attendance_rate", 100) < 60:
            factors.append("Critically low attendance (<60%)")
        if data.get("study_hours_per_week", 10) < 5:
            factors.append("Insufficient study hours (<5 hrs/week)")
        if data.get("assignment_completion", 100) < 70:
            factors.append("Low assignment submission rate (<70%)")
        if data.get("midterm_score", 100) < 40:
            factors.append("Poor midterm performance (<40%)")
        if data.get("previous_gpa", 4.0) < 2.0:
            factors.append("Low historical GPA (<2.0)")
        return factors

    def _fallback_predict(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Rule-based fallback when models aren't loaded yet."""
        score = (
            data.get("attendance_rate", 0) * 0.20
            + data.get("assignment_avg", 0) * 0.25
            + data.get("midterm_score", 0) * 0.25
            + data.get("study_hours_per_week", 0) * 1.5
            + data.get("previous_gpa", 0) * 5
            + data.get("quiz_avg", 0) * 0.10
            + data.get("assignment_completion", 0) * 0.10
            + data.get("participation_score", 0) * 1.0
        )
        if score >= 70:
            pred = "Pass"
        elif score >= 45:
            pred = "At Risk"
        else:
            pred = "Fail"
        return {
            "prediction": pred,
            "confidence": 0.70,
            "probabilities": {"Pass": 0.0, "At Risk": 0.0, "Fail": 0.0},
            "suggestions": self._generate_suggestions(data, pred, {f: 1/8 for f in FEATURE_COLUMNS}),
            "risk_factors": self._identify_risk_factors(data),
            "feature_importances": {f: 1/8 for f in FEATURE_COLUMNS},
            "color": PERFORMANCE_COLORS.get(pred, "#6c757d"),
            "icon": PERFORMANCE_ICONS.get(pred, "❓"),
        }

