import pandas as pd

# Load the cleaned dataset
df_model = pd.read_csv("OnlineRetail_cleaned.csv")  # Update path if needed

# Ensure 'CustomerGroup' column exists
if 'CustomerGroup' not in df_model.columns:
    df_model['CustomerGroup'] = pd.cut(df_model['Quantity'], bins=[0, 10, 100], labels=['Low', 'High'])

import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Load the trained model
model = joblib.load("customer_segmentation_model.pkl")

st.title("Customer Segmentation Predictor")

quantity = st.number_input(
    "Enter Quantity Purchased:",
    min_value=1,
    step=1,
    key="quantity_input"  # Unique key for this number input
)

unit_price = st.number_input(
    "Enter Unit Price:",
    min_value=0.0,
    step=0.01,
    key="unit_price_input"  # Unique key for this number input
)

if st.button("Predict", key="predict_button_1"):
    result = model.predict([[quantity, unit_price]])
    st.write(f"Predicted Customer Group: {result[0]}")

if st.button("Predict Again", key="predict_button_2"):
    result = model.predict([[quantity, unit_price]])
    st.write(f"Predicted Customer Group: {result[0]}")
    
    
    
#----------------------------------------
    
#import streamlit as st
#import joblib
#import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd

# Load trained model
model = joblib.load("customer_segmentation_model.pkl")

st.title("Online Retail Analysis App")

# ✅ Feature Importance Plot
st.subheader("Feature Importance Plot")
feature_importance = model.feature_importances_
plt.bar(["Quantity", "UnitPrice"], feature_importance, color='skyblue')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
st.pyplot(plt)

# ✅ Customer Group Distribution Plot
st.subheader("Customer Group Distribution")
group_counts = df_model['CustomerGroup'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values, palette='viridis')
plt.title("Customer Group Distribution")
plt.xlabel("Customer Group")
plt.ylabel("Count")
st.pyplot(plt)

# 🎯 User Input for Predictions (Make sure this section follows after visuals)
quantity = st.number_input("Enter Quantity Purchased:", min_value=1, step=1)
unit_price = st.number_input("Enter Unit Price:", min_value=0.0, step=0.01)

if st.button("Predict"):
    result = model.predict([[quantity, unit_price]])
    st.write(f"Predicted Customer Group: {result[0]}")



