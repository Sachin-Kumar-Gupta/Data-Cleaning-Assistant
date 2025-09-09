import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import io

st.set_page_config(page_title="AI Data Cleaning Assistant", layout="wide")

st.title("🧹 AI Data Cleaning Assistant by Sachin Gupta")
st.markdown("Upload your dataset and clean it step by step 🚀")

# -------------------------------
# 1. Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())

    # Summary statistics
    st.dataframe(df.head())
    
    st.header("📊 Dataset Overview")
    
    if df is not None:
        st.subheader("Quick Summary (Numeric Columns)")
        st.dataframe(df.describe())
    
        st.subheader("Quick Summary (Categorical Columns)")
        st.dataframe(df.describe(include='object'))
    
        st.subheader("Dataset Info")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        st.subheader("Duplicate Rows")
        st.write(f"**Your dataset contain  {df.duplicated().sum()} duplicates rows**")


    # -------------------------------
    # 2. Missing Values Handling
    # -------------------------------
    st.subheader("🕳 Missing Value Analysis")

    missing_report = df.isnull().sum().reset_index()
    missing_report.columns = ["Column", "Missing Values"]
    missing_report["% Missing"] = (missing_report["Missing Values"] / len(df)) * 100

    st.dataframe(missing_report)

    # Let user pick column
    col_choice = st.selectbox("Select a column to handle missing values:", df.columns)

    if df[col_choice].isnull().sum() > 0:
        st.write(f"⚠️ Column **{col_choice}** has {df[col_choice].isnull().sum()} missing values.")
        
        # -------------------------------
        # Visualization for Numeric Columns
        # -------------------------------
        if np.issubdtype(df[col_choice].dtype, np.number):
            st.subheader(f"📈 Distribution of {col_choice}")
    
            fig, ax = plt.subplots(figsize=(6,4))  # smaller plot
            sns.histplot(df[col_choice], kde=True, bins=30, ax=ax, color="skyblue")
    
            mean_val = round(df[col_choice].mean(skipna=True), 2)
            median_val = round(df[col_choice].median(skipna=True), 2)
    
            ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.2f}")
            ax.axvline(median_val, color="green", linestyle="-.", label=f"Median = {median_val:.2f}")
            ax.legend()
            
            # 🔽 Smaller font sizes
            ax.set_xlabel(col_choice, fontsize=6)
            ax.set_ylabel("Frequency", fontsize=6)
            ax.tick_params(axis='both', labelsize=3)  
            ax.legend(fontsize=4)
    
            st.pyplot(fig)
            st.write(f"📊 Skewness = {df[col_choice].skew():.2f}")
    
            default_method = "Median" if df[col_choice].skew() > 1 else "Mean"
            st.write(f"👉 Suggested method: **{default_method} imputation**")
            method = st.radio("Choose method:", ["Mean", "Median", "Forward Fill", "Backward Fill", "Drop Rows"])
    
        # -------------------------------
        # Visualization for Categorical Columns
        # -------------------------------
        else:
            st.subheader(f"📊 Category Distribution of {col_choice}")
    
            fig, ax = plt.subplots(figsize=(6,4))
            df[col_choice].value_counts().plot(kind="bar", ax=ax, color="skyblue")
            ax.set_ylabel("Frequency")
            # 🔽 Smaller font sizes
            ax.set_xlabel(col_choice, fontsize=6)
            ax.set_ylabel("Frequency", fontsize=6)
            ax.tick_params(axis='both', labelsize=3)  
            ax.legend(fontsize=4)

            st.pyplot(fig)
    
            st.write(f"👉 Mode (most frequent value): {df[col_choice].mode()[0]}")
            method = st.radio("Choose method:", ["Mode", "Unknown Category", "Forward Fill", "Backward Fill", "Drop Rows"])

        # Suggest method
        if np.issubdtype(df[col_choice].dtype, np.number):
            default_method = "Median" if df[col_choice].skew() > 1 else "Mean"
            st.write(f"👉 Suggested method: **{default_method} imputation**")
            #method = st.radio("Choose method:", ["Mean", "Median", "Forward Fill", "Backward Fill", "Drop Rows"],key=f"impute_{col_choice}")
        else:
            st.write("👉 Suggested method: **Mode (most frequent) imputation**")
            #method = st.radio("Choose method:", ["Mode", "Unknown Category", "Forward Fill", "Backward Fill", "Drop Rows"],key=f"impute_{col_choice}")

        # Apply chosen method
        if st.button("Apply Imputation"):
            if method == "Mean":
                df[col_choice].fillna(df[col_choice].mean(), inplace=True)
            elif method == "Median":
                df[col_choice].fillna(df[col_choice].median(), inplace=True)
            elif method == "Mode":
                df[col_choice].fillna(df[col_choice].mode()[0], inplace=True)
            elif method == "Unknown Category":
                df[col_choice].fillna("Unknown", inplace=True)
            elif method == "Forward Fill":
                df[col_choice].fillna(method="ffill", inplace=True)
            elif method == "Backward Fill":
                df[col_choice].fillna(method="bfill", inplace=True)
            elif method == "Drop Rows":
                df.dropna(subset=[col_choice], inplace=True)

            st.success(f"✅ Missing values in **{col_choice}** handled using {method}!")
            st.dataframe(df.head())

    else:
        st.success(f"🎉 Column **{col_choice}** has no missing values.")

    # ----------------------------
    # 📌 Phase 2: Outlier Detection & Handling
    # ----------------------------
    st.header("📊 Outlier Detection & Handling")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_cols:
        col_choice = st.selectbox("Select a numeric column to check for outliers:", numeric_cols, key="outlier_col_select")
    
        if col_choice:
            st.subheader("⚙️ Outlier Definition Settings")
            method_choice = st.radio("Choose method:", ["IQR", "Z-Score", "Manual"], key=f"method_{col_choice}")

            if method_choice == "IQR":
                st.info("ℹ️ **Formula**: Lower = Q1 - k × IQR, Upper = Q3 + k × IQR")
                factor = st.slider("Set IQR factor:", 1.0, 3.0, 1.5, 0.1, key=f"iqr_{col_choice}")
                Q1 = df[col_choice].quantile(0.25)
                Q3 = df[col_choice].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                outliers = df[(df[col_choice] < lower_bound) | (df[col_choice] > upper_bound)]
            
            elif method_choice == "Z-Score":
                st.info("ℹ️ **Formula**: z = (x - mean) / std → |z| > threshold = outlier")
                threshold = st.slider("Set Z-score threshold:", 2.0, 5.0, 3.0, 0.1, key=f"zscore_{col_choice}")
                col_mean = df[col_choice].mean()
                col_std = df[col_choice].std()
                z_scores = (df[col_choice] - col_mean) / col_std
                outliers = df[(np.abs(z_scores) > threshold)]
                lower_bound, upper_bound = None, None
            
            else:  # Manual min-max
                st.markdown("✍️ Define your own thresholds")
                min_val = st.number_input("Set Minimum Value", float(df[col_choice].min()), float(df[col_choice].max()))
                max_val = st.number_input("Set Maximum Value", float(df[col_choice].min()), float(df[col_choice].max()), value=float(df[col_choice].max()))
                outliers = df[(df[col_choice] < min_val) | (df[col_choice] > max_val)]
                lower_bound, upper_bound = min_val, max_val
    
            st.write(f"Detected **{len(outliers)} outliers** in `{col_choice}` using {method_choice} method.")
    
            # Boxplot
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(x=df[col_choice], ax=ax, color="skyblue")
            ax.set_title(f"Boxplot of {col_choice}", fontsize=10)
            st.pyplot(fig)
    
            # Histogram
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(df[col_choice], kde=True, bins=30, ax=ax, color="lightgreen")
            if method_choice == "IQR":
                ax.axvline(lower_bound, color="red", linestyle="--", label=f"Lower Bound ({lower_bound:.2f})")
                ax.axvline(upper_bound, color="red", linestyle="--", label=f"Upper Bound ({upper_bound:.2f})")
                ax.legend(fontsize=8)
            ax.set_title(f"Distribution of {col_choice}", fontsize=10)
            st.pyplot(fig)
    
            if not outliers.empty:
                st.subheader("🔎 Outlier Rows Preview")
                st.dataframe(outliers.head(10))
    
            # Handling Options with Apply button
            st.subheader("✂️ Handle Outliers")
            action = st.radio(
                "Choose an action:",
                ["Do Nothing", "Remove Outliers", "Cap Outliers (Winsorization)"],
                key=f"outlier_action_{col_choice}"
            )
    
            if st.button("Apply Outlier Handling", key=f"apply_outliers_{col_choice}"):
                if action == "Remove Outliers":
                    df[col_choice] = df[col_choice].apply(
                        lambda x: x if (lower_bound is None or lower_bound <= x <= upper_bound) else np.nan
                    )
                    st.success(f"Outliers removed from `{col_choice}` (set to NaN for later imputation).")
    
                elif action == "Cap Outliers (Winsorization)" and method_choice == "IQR":
                    df[col_choice] = np.where(df[col_choice] < lower_bound, lower_bound,
                                              np.where(df[col_choice] > upper_bound, upper_bound, df[col_choice]))
                    st.success(f"Outliers capped within [{lower_bound:.2f}, {upper_bound:.2f}] for `{col_choice}`.")
    
                else:
                    st.info("No changes applied.")

    # -------------------------------
    # 3. Download Cleaned Data
    # -------------------------------
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        label="📥 Download Current Cleaned CSV",
        data=buffer,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )