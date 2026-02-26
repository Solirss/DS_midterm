## Step 00 - Import of the packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import seaborn as sns


st.set_page_config(
    page_title="Califronia Housing Dashboard 🏡",
    layout="centered",
    page_icon="🏡",
)


## Step 01 - Setup
st.sidebar.title("California - Real Estate Agency 🏡")
page = st.sidebar.selectbox("Select Page",["Regression Model 📊"])

df = pd.read_csv("Student_data.csv")


# ---------------------------- Regression Part ------------------------------------------
if page == "Regression Model 📊":

# ---------------------------- Preprocessing for regression model -----------------------

    # - Change gender and major into numeric values
    df["Gender"] = df["Gender"].astype("category").cat.codes
    df["Major"] = df["Major"].astype("category").cat.codes

    # - Getting X and y for regression model
    from sklearn.model_selection import train_test_split
    X = df[['Gender',	'Age',	'Major',	'Attendance_Pct',	'Study_Hours_Per_Day',	'Previous_GPA',	'Sleep_Hours',	'Social_Hours_Week']]
    y = df["Final_CGPA"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # - Train the regression model
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # - Evaluate the regression model
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    prediction = lr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    mae = mean_absolute_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    # - Create a dataframe to display the coefficients of the regression model
    coef_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": lr.coef_
    })

# ---------------------------- Code for web app ------------------------------------------

    ## Step 03 - Regression Model
    st.subheader("Regression Model 📊")

   
    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap 🔥", "Regression Model Accuracy 📈", "Sweet Spot"])

    with tab1:
        st.subheader("Correlation Matrix")
        df_numeric = df.select_dtypes(include=np.number)

        fig_corr, ax_corr = plt.subplots(figsize=(18,14))

        # create the plot, in this case with seaborn 
        sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='coolwarm')
        
        ## render the plot in streamlit 
        st.pyplot(fig_corr)

    with tab2:
        st.subheader("Regression Model Accuracy")
        st.write(f"Root Mean Squared Error: {rmse}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"R-squared Score: {r2}")
        st.subheader("Regression Coefficients")
        st.dataframe(coef_df)

        st.subheader("Actual vs Predicted Values (Scatter Plot)")
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, prediction)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted (Test Set)")

        # perfect prediction line
        plt.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()])

        st.pyplot(plt.gcf())

with tab3:
    st.subheader("Sweet Spot")

    # Initialize session state
    if "study_hours" not in st.session_state:
        st.session_state.study_hours = 6
    if "sleep_hours" not in st.session_state:
        st.session_state.sleep_hours = 8
    if "social_hours" not in st.session_state:
        st.session_state.social_hours = 2

    total_hours = 24

    def get_max(key):
        """Max allowed for a slider = 24 minus the other two sliders' current values"""
        others = {k: st.session_state[k] for k in ["study_hours", "sleep_hours", "social_hours"] if k != key}
        return max(0, total_hours - sum(others.values()))

    study = st.slider(
        "Hours spent studying per day",
        0, get_max("study_hours"),
        st.session_state.study_hours,
        key="study_hours"
    )

    sleep = st.slider(
        "Hours spent sleeping per day",
        0, get_max("sleep_hours"),
        st.session_state.sleep_hours,
        key="sleep_hours"
    )

    social = st.slider(
        "Hours spent on social life per day",
        0, get_max("social_hours"),
        st.session_state.social_hours,
        key="social_hours"
    )

    remaining = total_hours - study - sleep - social
    st.info(f"⏰ Remaining unallocated hours: **{remaining}**")

    # Social hours per week (model expects weekly, slider is daily)
    social_weekly = social * 7

    # Get other feature values — use dataset medians as neutral defaults
    gender_val = df["Gender"].quantile(0.8)
    age_val = df["Age"].quantile(0.8)
    major_val = df["Major"].quantile(0.8)
    attendance_val = df["Attendance_Pct"].quantile(0.8)
    prev_gpa_val = df["Previous_GPA"].quantile(0.8)

    # Build input array matching model's feature order
    input_data = pd.DataFrame([[
        gender_val,
        age_val,
        major_val,
        attendance_val,
        study,          # Study_Hours_Per_Day
        prev_gpa_val,
        sleep,          # Sleep_Hours
        social_weekly   # Social_Hours_Week
    ]], columns=['Gender', 'Age', 'Major', 'Attendance_Pct',
                 'Study_Hours_Per_Day', 'Previous_GPA', 'Sleep_Hours', 'Social_Hours_Week'])

    predicted_cgpa = lr.predict(input_data)[0]

    # Top 10% threshold from actual data
    top_10_threshold = df["Final_CGPA"].quantile(0.90)

    st.metric("📊 Predicted CGPA", f"{predicted_cgpa:.2f}")




