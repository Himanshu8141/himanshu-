import pandas as pd

# Load dataset
df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")

# Step 1: Check missing values
print("Missing Values Before Cleaning:")
print(df.isnull().sum())

# Step 2: Drop rows where CustomerID is missing
df_cleaned = df.dropna(subset=['CustomerID'])

# Step 3: Remove duplicate rows
df_cleaned = df_cleaned.drop_duplicates()

# Step 4: Convert InvoiceDate column to datetime format
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])

# Verify changes
print("\nMissing Values After Cleaning:")
print(df_cleaned.isnull().sum())

print("\nData Types After Cleaning:")
print(df_cleaned.dtypes)
# Save cleaned dataset to CSV file
df_cleaned.to_csv("OnlineRetail_cleaned.csv", index=False)

print("✅ Cleaned dataset saved as 'OnlineRetail_cleaned.csv'")