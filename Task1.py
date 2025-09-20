Idea: Exploratory Data Analysis (EDA) on Retail Sales Data


Description:

In this project, you will work with a dataset containing information about retail sales. The goal is
to perform exploratory data analysis (EDA) to uncover patterns, trends, and insights that can
help the retail business make informed decisions.

# --- Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Load Dataset ---
file_path = "retail_sales_dataset.csv"   # Update path if needed(I used Dataset 1)
df = pd.read_csv(file_path)

# --- Data Cleaning ---
# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop duplicates
df.drop_duplicates(inplace=True)

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# --- Descriptive Statistics ---
desc_stats = {
    'Mean': df[['Quantity', 'Price per Unit', 'Total Amount']].mean(),
    'Median': df[['Quantity', 'Price per Unit', 'Total Amount']].median(),
    'Mode': df[['Quantity', 'Price per Unit', 'Total Amount']].mode().iloc[0],
    'Standard Deviation': df[['Quantity', 'Price per Unit', 'Total Amount']].std()
}
desc_stats_df = pd.DataFrame(desc_stats)
print("\nDescriptive Statistics:\n", desc_stats_df)

# --- Time Series Analysis ---
sales_trend = df.groupby('Date')['Total Amount'].sum().resample('M').sum()

plt.figure(figsize=(10,6))
sales_trend.plot(kind='line')
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()

# --- Customer & Product Analysis ---
# Sales by Gender
gender_sales = df.groupby('Gender')['Total Amount'].sum()
plt.figure(figsize=(6,4))
sns.barplot(x=gender_sales.index, y=gender_sales.values)
plt.title("Sales by Gender")
plt.show()

# Sales by Age Group
age_sales = df.groupby(pd.cut(df['Age'], bins=[18,25,35,45,55,65,100]))['Total Amount'].sum()
plt.figure(figsize=(8,5))
age_sales.plot(kind='bar')
plt.title("Sales by Age Group")
plt.show()

# Sales by Product Category
product_sales = df.groupby('Product Category')['Total Amount'].sum()
plt.figure(figsize=(8,5))
product_sales.plot(kind='bar', color='skyblue')
plt.title("Sales by Product Category")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# --- Recommendations ---
recommendations = [
    "Increase marketing campaigns around peak sales months to maximize revenue.",
    "Target younger age groups (18–35) with promotions, as they contribute significantly to sales.",
    "Promote high-performing product categories more aggressively.",
    "Balance marketing campaigns across genders since both contribute strongly to sales.",
    "Introduce bundle offers for frequently purchased-together products to increase basket size."
]

print("\nActionable Recommendations:")
for rec in recommendations:
    print("-", rec)
