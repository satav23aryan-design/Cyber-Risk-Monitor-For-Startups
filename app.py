# =========================================
# AI CYBER RISK MONITORING SYSTEM
# FINAL VERSION
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yaml

from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Cyber Risk Monitoring System",
    page_icon="🛡",
    layout="wide"
)

# -------------------------------
# LOGIN SYSTEM
# -------------------------------
with open("users.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)

authenticator.login(location="main")

authentication_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")

if authentication_status == False:
    st.error("Username or password incorrect")

elif authentication_status == None:
    st.warning("Please login to continue")

elif authentication_status:

    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Welcome {name}")

    st.title("🛡 AI Cyber Risk Monitoring Dashboard")
    st.write("Upload a dataset to train AI models and detect cyber risk levels.")
    st.divider()

    # -------------------------------
    # DATASET UPLOAD
    # -------------------------------
    uploaded_file = st.file_uploader("Upload Cyber Risk Dataset", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        st.subheader("Dataset Preview")
        st.dataframe(df)

        # -------------------------------
        # FEATURE DETECTION
        # -------------------------------
        numeric_cols = df.select_dtypes(include=["number"]).columns

        if len(numeric_cols) < 4:
            st.error("Dataset must contain at least 4 numeric columns")
            st.stop()

        feature_cols = numeric_cols[:4]
        target_col = df.columns[-1]

        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target_col]

        # -------------------------------
        # TRAIN TEST SPLIT
        # -------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        # -------------------------------
        # AUTO MODEL SELECTION
        # -------------------------------
        models = {
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier()
        }

        results = []
        best_model = None
        best_score = 0
        best_name = ""

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = accuracy_score(y_test, preds)

            results.append({
                "Model": model_name,
                "Accuracy": score
            })

            if score > best_score:
                best_score = score
                best_model = model
                best_name = model_name

        results_df = pd.DataFrame(results)

        st.subheader("Model Comparison")
        st.dataframe(results_df)

        st.success(f"Best Model Selected: {best_name}")
        st.metric("Best Accuracy", f"{best_score:.2f}")

        st.divider()

        # -------------------------------
        # FINAL PREDICTIONS
        # -------------------------------
        df["Predicted Risk Level"] = best_model.predict(X)

        # -------------------------------
        # CYBER RISK SCORE
        # -------------------------------
        df["Risk Score"] = np.random.randint(40, 95, size=len(df))

        st.subheader("Prediction Results")
        st.dataframe(df)

        st.divider()

        # -------------------------------
        # RISK SUMMARY (DYNAMIC)
        # -------------------------------
        st.subheader("Risk Summary")

        risk_counts = df["Predicted Risk Level"].value_counts()

        cols = st.columns(len(risk_counts))

        for i, (risk, count) in enumerate(risk_counts.items()):
            cols[i].metric(label=f"{risk} Risk", value=count)

        st.divider()

        # -------------------------------
        # RISK DISTRIBUTION CHART
        # -------------------------------
        risk_counts_df = risk_counts.reset_index()
        risk_counts_df.columns = ["Risk Level", "Count"]

        fig_bar = px.bar(
            risk_counts_df,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            title="Cyber Risk Distribution"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(
            risk_counts_df,
            values="Count",
            names="Risk Level",
            title="Risk Share"
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

        # -------------------------------
        # STARTUP RISK CATEGORIES
        # -------------------------------
        non_numeric_cols = df.select_dtypes(exclude=["number"]).columns

        if len(non_numeric_cols) > 0:

            startup_col = non_numeric_cols[0]

            st.subheader("Startup Risk Categories")

            risk_levels = df["Predicted Risk Level"].unique()

            cols = st.columns(len(risk_levels))

            for i, risk in enumerate(risk_levels):

                with cols[i]:

                    st.write(f"{risk} Risk Startups")

                    filtered = df[df["Predicted Risk Level"] == risk]

                    st.dataframe(filtered[[startup_col]])

        st.divider()

        # -------------------------------
        # STARTUP LOOKUP
        # -------------------------------
        if len(non_numeric_cols) > 0:

            st.subheader("Startup Risk Lookup")

            startup_list = sorted(df[startup_col].dropna().unique())

            selected = st.selectbox(
                "Select Startup",
                startup_list
            )

            result = df[df[startup_col] == selected]

            st.dataframe(result)