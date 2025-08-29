
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the path to the CSV file
file_path = r"C:\Users\user\Desktop\Sales and Customer Insights\Sample - Superstore.csv"

# Load the dataset
df = pd.read_csv(file_path, encoding='ISO-8859-1') #encoding='ISO-8859-1' fix problematic letters

# Display first few rows to understand structure
print("Sample data:")
print(df.head())

# Getting Information About Our Dataset
print("\nDetails:")
df.info()

#Top 5 Profitable Sub-Categories
subcat_profit = df.groupby('Sub-Category')['Profit'].sum()
top5_subcats = subcat_profit.sort_values(ascending=False).head(5)
print("\nTop 5 Sub-Categories by Profit:")
print(top5_subcats)
# Plot the result as a bar chart
plt.figure(figsize=(10, 6))
top5_subcats.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Top 5 Sub-Categories by Total Profit')
plt.xlabel('Sub-Category')
plt.ylabel('Total Profit ($)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#What is the total sales and profit for each region in 2017?
df['Order Date'] = pd.to_datetime(df['Order Date']) #Convert 'Order Date' to datetime format
df_2017 = df[df['Order Date'].dt.year == 2017] # Filter data for the year 2017
region_summary = df_2017.groupby('Region')[['Sales', 'Profit']].sum().sort_values(by='Sales', ascending=False) #Group by Region and calculate total Sales and Profit
print("\nRegional Sales and Profit in 2017:")
print(region_summary)

#Which products had more than $5,000 in sales but generated a loss?
loss_products = df[(df['Sales'] > 5000) & (df['Profit'] < 0)]
unique_loss_products = loss_products[['Product Name', 'Sales', 'Profit']]
print("\nSample of High-Sales but Unprofitable Products:")
print(unique_loss_products)

#Compare monthly sales trends for 'Furniture' vs. 'Technology'
category_df = df[df['Category'].isin(['Furniture', 'Technology'])].copy() # Filter data for Furniture and Technology only
category_df['Month'] = category_df['Order Date'].dt.to_period('M') # Create a 'Month' column.dt.to_period remain just the month and the year
monthly_trends = category_df.groupby(['Month', 'Category'])['Sales'].sum().unstack() #Group by Month and Category, then sum Sales for each category
# Plot line chart
monthly_trends.plot(kind='line', figsize=(12, 6), marker='o')
plt.title('Monthly Sales Trend: Furniture vs. Technology')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.grid(True)
plt.legend(title='Category')
plt.tight_layout()
plt.show()

#Sales by Category and Ship Mode
shipmode_matrix = df.groupby(['Category', 'Ship Mode'])['Sales'].sum().unstack()
# Plot
shipmode_matrix.plot(kind='bar', figsize=(10, 6), colormap='Set1')
plt.title('Sales by Category and Ship Mode')
plt.xlabel('Category')
plt.ylabel('Sales ($)')
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.legend(title='Ship Mode')
plt.tight_layout()
plt.show()

# Sales Distribution by Product Category
category_sales = df.groupby('Category')['Sales'].sum()
plt.figure(figsize=(8, 6))
plt.pie(category_sales,
        labels=category_sales.index,  # Labels around the pie (category names)
        autopct='%1.1f%%',  # Show percentage format inside slices
        startangle=140,  # Rotate chart for better angle presentation
        )
plt.title('Sales Distribution by Product Category')
plt.axis('equal')  # Keeps pie as a circle
plt.tight_layout() #Automatically adjust layout to avoid label cut-off
plt.show()

#NumPy
# Extract Sales column as NumPy array Using Statistic
print("---Statistic and NumPy---")
sales_array = df['Sales'].values
# Calculate mean and std using NumPy
mean_sales = np.mean(sales_array)
std_sales = np.std(sales_array) #Standard Deviation= (mean((x−x.mean())))^0.5
z_scores = (sales_array - mean_sales) / std_sales # Compute Z-score
df['Sales_Z_Score'] = z_scores # Add to DataFrame
df['Sales_Outlier'] = np.abs(df['Sales_Z_Score']) > 3 # Flag high outliers (Z > 3)
print("Top Sales Outliers:") # Show sample of outliers
print(df[df['Sales_Outlier']][['Order ID', 'Sales', 'Sales_Z_Score']].head(3))

# Convert Profit column to NumPy array
profit_array = df['Profit'].values
profit_min = np.min(profit_array)
profit_max = np.max(profit_array)
profit_normalized = (profit_array - profit_min) / (profit_max - profit_min)
df['Profit_Normalized'] = profit_normalized # Add back to DataFrame
print("\nNormalized Profit Sample:")# Show sample
print(df[['Profit', 'Profit_Normalized']].head())


