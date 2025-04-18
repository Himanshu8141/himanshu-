import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df_cleaned = pd.read_csv("OnlineRetail_cleaned.csv")

# Verify dataset loading
print(df_cleaned.head())  # This displays the first few rows of the dataset#step1: import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visuals
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visuals
sns.set_style("darkgrid")


#Step 2: Transaction Trends Over Time
# Extract year and month from InvoiceDate
# Convert InvoiceDate to datetime format
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])

# Extract year and month from InvoiceDate
df_cleaned['YearMonth'] = df_cleaned['InvoiceDate'].dt.to_period('M')

# Aggregate sales by month
monthly_sales = df_cleaned.groupby('YearMonth')['Quantity'].sum()

# Plot monthly sales trend
plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line', marker='o', color='blue')
plt.xlabel("Year-Month")
plt.ylabel("Total Quantity Sold")
plt.title("Monthly Sales Trend")
plt.xticks(rotation=45)
plt.show()


#Step 3: Top-Selling Products
# Aggregate product sales
top_products = df_cleaned.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

# Plot top-selling products
plt.figure(figsize=(12, 6))
sns.barplot(x=top_products.values, y=top_products.index, palette='coolwarm')
plt.xlabel("Quantity Sold")
plt.ylabel("Product Description")
plt.title("Top-Selling Products")
plt.show()

#Step 4: Customer Purchase Behavior
# Aggregate customer purchases
top_customers = df_cleaned.groupby('CustomerID')['Quantity'].sum().sort_values(ascending=False).head(10)

# Plot top customers
plt.figure(figsize=(12, 6))
sns.barplot(x=top_customers.index.astype(int), y=top_customers.values, palette='viridis')
plt.xlabel("Customer ID")
plt.ylabel("Total Quantity Purchased")
plt.title("Top 10 Customers by Purchase Volume")
plt.xticks(rotation=45)
plt.show()

#Step 5: Correlation Analysis
# Compute correlation matrix
corr_matrix = df_cleaned[['Quantity', 'UnitPrice', 'CustomerID']].corr()

# Plot heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()