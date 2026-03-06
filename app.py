"""
Streamlit Student Performance Predictor App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from model.predictor import (
    StudentPredictor, FEATURE_COLUMNS, FEATURE_LABELS,
    PERFORMANCE_COLORS, PERFORMANCE_ICONS,
)

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_predictor():
    return StudentPredictor()

predictor = load_predictor()

# Sidebar
with st.sidebar:
    st.markdown("## Settings")
    model_choice = st.selectbox(
        "Prediction Model",
        options=["random_forest", "gradient_boosting", "ensemble"],
        format_func=lambda x: {
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
            "ensemble": "Ensemble (Both)",
        }[x],
    )
    st.markdown("---")
    st.markdown("### Model Performance")
    if predictor.metadata:
        metrics = predictor.metadata.get("metrics", {})
        for m_name, m_label in [("random_forest", "Random Forest"), ("gradient_boosting", "Gradient Boosting")]:
            m = metrics.get(m_name, {})
            if m:
                st.markdown(f"**{m_label}**")
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
                col2.metric("AUC", f"{m.get('auc', 0):.3f}")
    else:
        st.info("Train models first:\npython model/train_model.py")

# Main UI
st.markdown("# Student Performance Predictor")
st.markdown("Predict academic outcomes using attendance, assignments & study habits.")

tab1, tab2, tab3 = st.tabs(["Predict", "Batch Analysis", "Feature Insights"])

# ---- Tab 1: Single Prediction ----
with tab1:
    st.markdown("### Enter Student Metrics")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### Engagement")
        attendance = st.slider("Attendance Rate (%)", 0, 100, 75)
        study_hours = st.slider("Study Hours / Week", 0.0, 20.0, 8.0, step=0.5)
        participation = st.slider("Participation Score (0-10)", 0.0, 10.0, 5.0, step=0.5)
    with col_b:
        st.markdown("#### Assessments")
        assignment_avg = st.slider("Assignment Average (%)", 0, 100, 70)
        assignment_completion = st.slider("Assignment Completion (%)", 0, 100, 80)
        quiz_avg = st.slider("Quiz Average (%)", 0, 100, 65)
    with col_c:
        st.markdown("#### Performance")
        midterm = st.slider("Midterm Score (%)", 0, 100, 65)
        prev_gpa = st.slider("Previous GPA (0.0-4.0)", 0.0, 4.0, 2.5, step=0.1)

    st.markdown("---")
    predict_btn = st.button("Predict Performance", type="primary", use_container_width=True)

    if predict_btn:
        student_data = {
            "attendance_rate": float(attendance),
            "assignment_avg": float(assignment_avg),
            "midterm_score": float(midterm),
            "study_hours_per_week": float(study_hours),
            "participation_score": float(participation),
            "previous_gpa": float(prev_gpa),
            "assignment_completion": float(assignment_completion),
            "quiz_avg": float(quiz_avg),
        }
        with st.spinner("Analysing student profile..."):
            if model_choice == "ensemble":
                all_results = predictor.predict_both(student_data)
                result = {
                    "prediction": all_results["ensemble"]["prediction"],
                    "confidence": all_results["ensemble"]["confidence"],
                    "probabilities": all_results["ensemble"]["probabilities"],
                    "color": all_results["ensemble"]["color"],
                    "icon": all_results["ensemble"]["icon"],
                    "suggestions": all_results["random_forest"]["suggestions"],
                    "risk_factors": all_results["random_forest"]["risk_factors"],
                    "feature_importances": all_results["random_forest"]["feature_importances"],
                }
            else:
                result = predictor.predict(student_data, model_choice)

        st.markdown("---")
        st.markdown("### Prediction Result")
        res_col1, res_col2, res_col3 = st.columns([1, 1, 2])
        with res_col1:
            st.metric("Prediction", f"{result['icon']} {result['prediction']}")
        with res_col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        with res_col3:
            if result["risk_factors"]:
                st.warning("Risk Factors: " + ", ".join(result["risk_factors"]))
            else:
                st.success("No critical risk factors detected.")

        # Probability chart
        probs = result["probabilities"]
        if any(v > 0 for v in probs.values()):
            fig_probs = go.Figure()
            colors = [PERFORMANCE_COLORS.get(cls, "#6c757d") for cls in probs]
            fig_probs.add_trace(go.Bar(
                x=list(probs.keys()),
                y=[v * 100 for v in probs.values()],
                marker_color=colors,
                text=[f"{v:.1%}" for v in probs.values()],
                textposition="outside",
            ))
            fig_probs.update_layout(
                title="Class Probability Distribution",
                yaxis_title="Probability (%)",
                yaxis_range=[0, 110],
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_probs, use_container_width=True)

        # Feature importance radar
        imps = result["feature_importances"]
        if imps:
            labels = [FEATURE_LABELS.get(k, k) for k in imps]
            values = list(imps.values())
            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor="rgba(31,78,121,0.2)",
                line_color="#1f4e79",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(values)])),
                title="Feature Importance Radar",
                height=380,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Suggestions
        st.markdown("### Personalised Improvement Suggestions")
        for s in result["suggestions"]:
            curr = f" (Current: {s['current']}, Target: {s['target']})" if s.get("current") is not None else ""
            st.info(f"{s.get('icon','')} **{s['feature']}**{curr}\n{s['message']} [Priority: {s.get('priority','—')}]")

# ---- Tab 2: Batch Analysis ----
with tab2:
    st.markdown("### Batch Student Analysis")
    use_sample = st.checkbox("Use sample dataset", value=True)
    uploaded_file = None if use_sample else st.file_uploader("Upload CSV", type="csv")

    if use_sample or uploaded_file:
        if use_sample:
            from model.train_model import generate_dataset
            df_batch = generate_dataset(n_samples=200, random_state=99)
            df_batch = df_batch.drop(columns=["performance"], errors="ignore")
        else:
            df_batch = pd.read_csv(uploaded_file)

        predictions = []
        for _, row in df_batch.iterrows():
            r = predictor.predict(row[FEATURE_COLUMNS].to_dict(), "random_forest")
            predictions.append(r["prediction"])
        df_batch["Predicted Performance"] = predictions

        dist = df_batch["Predicted Performance"].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("Pass", dist.get("Pass", 0))
        col2.metric("At Risk", dist.get("At Risk", 0))
        col3.metric("Fail", dist.get("Fail", 0))

        fig_pie = px.pie(
            values=dist.values, names=dist.index,
            color=dist.index,
            color_discrete_map=PERFORMANCE_COLORS,
            title="Predicted Performance Distribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_scatter = px.scatter(
            df_batch, x="attendance_rate", y="study_hours_per_week",
            color="Predicted Performance",
            color_discrete_map=PERFORMANCE_COLORS,
            size="assignment_avg",
            title="Attendance vs Study Hours (sized by Assignment Average)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.dataframe(df_batch[FEATURE_COLUMNS + ["Predicted Performance"]].head(50), use_container_width=True)

# ---- Tab 3: Feature Insights ----
with tab3:
    st.markdown("### Feature Distribution & Correlations")
    from model.train_model import generate_dataset
    df_insight = generate_dataset(n_samples=1000, random_state=7)

    feature_to_show = st.selectbox(
        "Select Feature",
        options=FEATURE_COLUMNS,
        format_func=lambda x: FEATURE_LABELS.get(x, x),
    )
    fig_dist = px.box(
        df_insight, x="performance", y=feature_to_show,
        color="performance",
        color_discrete_map=PERFORMANCE_COLORS,
        title=f"{FEATURE_LABELS.get(feature_to_show, feature_to_show)} by Performance Class",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    corr = df_insight[FEATURE_COLUMNS].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    avg_by_class = df_insight.groupby("performance")[FEATURE_COLUMNS].mean().reset_index()
    melted = avg_by_class.melt(
        id_vars="performance", value_vars=FEATURE_COLUMNS,
        var_name="Feature", value_name="Average"
    )
    fig_bar = px.bar(
        melted, x="Feature", y="Average", color="performance",
        barmode="group",
        color_discrete_map=PERFORMANCE_COLORS,
        title="Average Feature Values by Performance Class",
    )
    fig_bar.update_xaxes(tickangle=30)
    st.plotly_chart(fig_bar, use_container_width=True)

