import os
import pandas as pd

# Change working directory to the project folder
os.chdir("C:/Users/himan/OneDrive/Desktop/OnlineRetail_project")

# Load the dataset
df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")

# Display first few rows
print(df.head())