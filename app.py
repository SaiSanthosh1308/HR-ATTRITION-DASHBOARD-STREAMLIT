import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score
)

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide", page_icon="👥")

# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("HR_Attrition.csv")
    drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df

df = load_data()

st.title("👥 HR Attrition Dashboard")
st.caption(f"{len(df):,} employees — explore attrition patterns or train a model to predict flight risk.")

tab_explore, tab_predict = st.tabs(["🔍 Explore", "🤖 Predict"])

# =================================================================
# TAB 1: EXPLORE
# =================================================================
with tab_explore:
    st.sidebar.header("Filters")
    dept_sel = st.sidebar.multiselect("Department", sorted(df["Department"].unique()), default=sorted(df["Department"].unique()))
    gender_sel = st.sidebar.multiselect("Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
    jobrole_sel = st.sidebar.multiselect("Job Role", sorted(df["JobRole"].unique()), default=sorted(df["JobRole"].unique()))
    age_range = st.sidebar.slider("Age range", int(df["Age"].min()), int(df["Age"].max()),
                                   (int(df["Age"].min()), int(df["Age"].max())))
    attrition_sel = st.sidebar.multiselect("Attrition", sorted(df["Attrition"].unique()), default=sorted(df["Attrition"].unique()))

    fdf = df[
        (df["Department"].isin(dept_sel)) &
        (df["Gender"].isin(gender_sel)) &
        (df["JobRole"].isin(jobrole_sel)) &
        (df["Age"].between(*age_range)) &
        (df["Attrition"].isin(attrition_sel))
    ]

    st.markdown(f"**Showing {len(fdf):,} of {len(df):,} employees**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", len(fdf))
    attr_pct = (fdf["Attrition"].eq("Yes").mean() * 100) if len(fdf) else 0
    c2.metric("Attrition Rate", f"{attr_pct:.1f}%")
    c3.metric("Avg Monthly Income", f"${fdf['MonthlyIncome'].mean():,.0f}" if len(fdf) else "—")
    c4.metric("Avg Years at Company", f"{fdf['YearsAtCompany'].mean():.1f}" if len(fdf) else "—")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition by Department")
        dept_attr = fdf.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
        fig = px.bar(dept_attr, x="Department", y="Count", color="Attrition", barmode="group",
                     color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"})
        st.plotly_chart(fig, width='stretch')

        st.subheader("Age Distribution by Attrition")
        fig2 = px.histogram(fdf, x="Age", color="Attrition", barmode="overlay", nbins=20, opacity=0.7,
                             color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"})
        st.plotly_chart(fig2, width='stretch')

    with col2:
        st.subheader("Monthly Income vs Attrition")
        fig3 = px.box(fdf, x="Attrition", y="MonthlyIncome", color="Attrition",
                       color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"})
        st.plotly_chart(fig3, width='stretch')

        st.subheader("Attrition by Job Role")
        role_attr = fdf.groupby(["JobRole", "Attrition"]).size().reset_index(name="Count")
        fig4 = px.bar(role_attr, x="Count", y="JobRole", color="Attrition", orientation="h",
                       color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"})
        st.plotly_chart(fig4, width='stretch')

    st.subheader("Job Satisfaction vs Work-Life Balance (bubble = headcount)")
    bubble = fdf.groupby(["JobSatisfaction", "WorkLifeBalance", "Attrition"]).size().reset_index(name="Count")
    fig5 = px.scatter(bubble, x="JobSatisfaction", y="WorkLifeBalance", size="Count", color="Attrition",
                       color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"})
    st.plotly_chart(fig5, width='stretch')

    st.subheader("Correlation Heatmap (numeric features)")
    numeric_cols = fdf.select_dtypes(include=[np.number]).columns.tolist()
    corr = fdf[numeric_cols].corr()
    fig6 = px.imshow(corr, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig6, width='stretch')

    with st.expander("View filtered raw data"):
        st.dataframe(fdf, width='stretch')

# =================================================================
# TAB 2: PREDICT
# =================================================================
with tab_predict:
    st.subheader("Train an Attrition Prediction Model")

    left, right = st.columns([1, 2])
    with left:
        model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random seed", value=42, step=1)
        train_btn = st.button("Train model", type="primary")

    model_df = df.copy()
    model_df["target"] = model_df["Attrition"].map({"Yes": 1, "No": 0})
    model_df = model_df.drop(columns=["Attrition"])
    cat_cols = model_df.select_dtypes(include="object").columns.tolist()
    model_df = pd.get_dummies(model_df, columns=cat_cols)

    feature_cols = [c for c in model_df.columns if c != "target"]
    X = model_df[feature_cols]
    y = model_df["target"]

    if "trained" not in st.session_state:
        st.session_state.trained = False

    if train_btn:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state), stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=int(random_state))
        else:
            model = RandomForestClassifier(n_estimators=300, random_state=int(random_state))

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        st.session_state.trained = True
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_cols = feature_cols
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.y_proba = y_proba
        st.session_state.model_choice = model_choice

    if st.session_state.trained:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        y_proba = st.session_state.y_proba

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        with right:
            st.markdown(f"**Model: {st.session_state.model_choice}**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.2%}")
            m2.metric("Precision", f"{prec:.2%}")
            m3.metric("Recall", f"{rec:.2%}")
            m4.metric("F1 Score", f"{f1:.2%}")

        st.divider()
        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                x=["Predicted Stay", "Predicted Leave"],
                                y=["Actual Stay", "Actual Leave"],
                                color_continuous_scale="Blues")
            st.plotly_chart(fig_cm, width='stretch')

        with col_roc:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, width='stretch')

        if st.session_state.model_choice == "Random Forest":
            st.subheader("Top Feature Importances")
            importances = pd.Series(st.session_state.model.feature_importances_, index=st.session_state.feature_cols)
            importances = importances.sort_values(ascending=False).head(15).sort_values()
            fig_imp = px.bar(importances, orientation="h")
            fig_imp.update_layout(showlegend=False, yaxis_title="", xaxis_title="Importance")
            st.plotly_chart(fig_imp, width='stretch')
    else:
        with right:
            st.info("Click **Train model** on the left to get started.")
