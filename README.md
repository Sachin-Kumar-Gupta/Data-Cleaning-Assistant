# Automate Data Cleaning App Created by Sachin Gupta

**Interactive Streamlit App for Data Cleaning and Exploratory Data Analysis**

---

## 🔥 Overview

This project is a **user-friendly, Data cleaning tool** built with Python and Streamlit.  
It allows users to upload any dataset and perform **professional-level cleaning and analysis**, including:  

- Missing value detection and imputation  
- Outlier detection and handling (IQR, Z-score, or custom thresholds)  
- Data visualization with boxplots and distribution plots  
- Preview of dataset statistics for both numeric and categorical columns  
- Interactive controls for choosing cleaning methods  
- Apply changes in a controlled manner with an **Apply button**  

This app is designed to demonstrate **real-world data cleaning workflows** for a data science portfolio.

---

## 🚀 Features

1. **Dataset Overview**  
   - Quick summary using `df.describe()` for numeric and categorical columns  
   - Dataset info: column types and null values  

2. **Missing Values Handling**  
   - Column-wise selection  
   - Visualizations (histogram + boxplot)  
   - Imputation options: Mean, Median, Mode, Forward/Backward Fill, Drop Rows  
   - Apply button for controlled changes  

3. **Outlier Detection & Handling**  
   - Methods: IQR, Z-Score, Manual thresholds  
   - Adjustable sliders / inputs for thresholds  
   - Boxplot and distribution plot with outlier visualization  
   - Preview of outlier rows  
   - Handling options: Remove, Cap (Winsorization), Keep as is  
   - Apply button for controlled application  
   - Info icons to explain formulas  

4. **Professional UI**  
   - Compact plots and consistent rounding  
   - Column-wise unique keys to avoid Streamlit conflicts  
   - Flexible, interactive design for portfolio showcase  

---

## 🛠 Tech Stack

- **Python 3.10+**  
- **Streamlit** – for web-based interactive UI  
- **Pandas** – data manipulation and cleaning  
- **NumPy** – numerical operations  
- **Matplotlib & Seaborn** – visualization 

---

