# demo-practice
This is my first git repositary for practice and analysis my skills 

# Project 1: Retail Sales EDA

## retail_sales_eda.ipynb (Python Notebook Template)

"""
Retail Sales EDA — Ready-to-Upload Template
"""

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
df = pd.read_csv('retail_sales.csv')
df.head()

# 3. Basic Checks
print(df.info())
print(df.describe())

# 4. Missing Values
sns.heatmap(df.isnull(), cbar=False)

# 5. Univariate Analysis
sns.histplot(df['Sales'])

# 6. Bivariate Analysis
sns.scatterplot(x='Quantity', y='Sales', data=df)

# 7. Insights Summary
print("Key insights:")
print("- Insight 1...")
print("- Insight 2...")


## README.md
# Retail Sales EDA
- Complete exploratory data analysis of retail dataset.
- Techniques used: Missing values, outliers, correlation, trend analysis.
- Tools: Python, Pandas, Matplotlib, Seaborn.

---

# Project 2: House Price Prediction (Regression)

## house_price_prediction.ipynb
"""
House Price Prediction — ML Regression Model Template
"""

# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load Data
df = pd.read_csv('house_prices.csv')

# 3. Feature Selection
X = df[['Bedrooms', 'Bathrooms', 'Sqft']]
y = df['Price']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R2 Score:", r2)

## README.md
# House Price Prediction
- Built a regression model to predict house prices.
- Used Linear Regression + feature engineering.
- Metrics: MSE, R², RMSE.

---

# Project 3: Customer Segmentation (K-Means)

## kmeans_customer_segmentation.ipynb
"""
K-Means Customer Segmentation — Template
"""

# 1. Import
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 2. Load Data
df = pd.read_csv('customers.csv')

# 3. Select Features
X = df[['Age', 'Annual_Income', 'Spending_Score']]

# 4. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Find Optimal Clusters (Elbow)
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia)
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 6. Final Model
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 7. Plot Clusters
plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'])
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.show()

## README.md
# Customer Segmentation (K-Means)
- Applied K-Means clustering to segment customers.
- Used scaling + Elbow method + cluster visualization.
- Great for marketing + business segmentation.
