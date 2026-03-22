# Assignment 22 — Data Analysis and Machine Learning

## Overview

This assignment is divided into two sessions:

* **AM Session** → Focus on Data Analysis using Pandas and Visualization (Diamonds Dataset)
* **PM Session** → Focus on Machine Learning (Regression & Classification using data.csv)

The assignment demonstrates practical understanding of:

* Data selection and filtering
* Statistical analysis
* Data visualization
* Supervised machine learning
* Model evaluation

---

# AM SESSION — DATA ANALYSIS (Diamonds Dataset)

## Part A — Concept Application

### 1. Data Selection using Pandas

**Code:**

```python
df.loc[0:4, ['carat', 'price']]
df.loc[10:15, ['cut', 'color']]
df.loc[:, ['price']]

df.iloc[0:5, 0:2]
df.iloc[10:20, 2:5]
df.iloc[0]
```

**Output:**

* Displays selected rows and columns using both label-based and index-based methods.

**Explanation:**

* `loc` → label-based indexing (column names)
* `iloc` → index-based indexing (positions)

---

### 2. Filtering Data

**Code:**

```python
df[(df['price'] > 5000) & (df['carat'] > 1)]
df[df['cut'] == 'Premium']
df[df['price'] > 10000]
```

**Output:**

* Filtered subsets of diamonds based on conditions.

**Explanation:**

* Helps extract meaningful subsets like high-value diamonds or specific categories.

---

### 3. Descriptive Statistics

**Code:**

```python
df.describe()
```

**Output:**

* Summary statistics including:

  * Mean
  * Standard deviation
  * Minimum and maximum values
  * Quartiles

**Explanation:**

* Useful for understanding distribution and detecting outliers.

---

### 4. Histogram

**Code:**

```python
import matplotlib.pyplot as plt

plt.hist(df['price'], bins=30)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Price Distribution")
plt.show()
```

**Output:**

* Histogram showing distribution of diamond prices.

**Explanation:**

* Right-skewed distribution → many low-price diamonds, few high-price ones.

---

### 5. Bar Plot

**Code:**

```python
import seaborn as sns

sns.barplot(x='cut', y='price', data=df)
plt.title("Average Price by Cut")
plt.show()
```

**Output:**

* Bar chart comparing average price across cut categories.

**Explanation:**

* Helps identify which cut category has higher average price.

---

### 6. Line Chart

**Code:**

```python
df['price'].head(100).plot()
plt.xlabel("Index")
plt.ylabel("Price")
plt.title("Price Trend")
plt.show()
```

**Output:**

* Line plot of price variation.

**Explanation:**

* Shows fluctuations in price without a clear trend.

---

### 7. KDE Plot

**Code:**

```python
sns.kdeplot(df['price'])
plt.title("Density Plot")
plt.show()
```

**Output:**

* Smooth density curve of price distribution.

**Explanation:**

* Confirms skewness and distribution shape.

---

## Part B — Stretch Problem

### 1. Grouped Analysis

**Code:**

```python
grouped = df.groupby('cut')['price'].mean()
grouped
```

**Output:**

* Average price for each cut category.

**Explanation:**

* Groups data and computes mean for comparison.

---

### 2. Bar Plot (Grouped Data)

**Code:**

```python
grouped.plot(kind='bar')
plt.xlabel("Cut")
plt.ylabel("Average Price")
plt.title("Average Price by Cut")
plt.show()
```

**Output:**

* Bar chart comparing average prices.

---

### 3. Compare Two Numerical Features

**Option A — Line Chart**

```python
df[['carat', 'price']].head(100).plot()
plt.title("Carat vs Price Trend")
plt.show()
```

**Option B — KDE Plot**

```python
sns.kdeplot(df['carat'], label='Carat')
sns.kdeplot(df['price'], label='Price')
plt.legend()
plt.title("Carat vs Price Distribution")
plt.show()
```

---

### 4. Insights

* Different cut categories have different average prices
* Price has a wider spread than carat
* Carat values are more concentrated
* Price distribution is skewed
* Higher carat generally leads to higher price

---

## Part C — Interview Ready

### Q1 — Difference between loc and iloc

* `loc` → label-based indexing
* `iloc` → position-based indexing

---

### Q2 — Filter rows greater than average

**Code:**

```python
avg_price = df['price'].mean()
df[df['price'] > avg_price]
```

**Output:**

* Rows where price is above average.

---

### Q3 — Purpose of describe()

* Provides statistical summary
* Helps understand:

  * Distribution
  * Spread
  * Outliers

---

## Part D — AI-Augmented Task

### Prompt

"Explain how to perform data analysis using Pandas and visualization using Matplotlib with examples."

### AI Output (Summary)

* Load data
* Clean data
* Analyze using Pandas
* Visualize using Matplotlib

### Evaluation

* Plots are correct
* Explanation is meaningful and beginner-friendly

---

# PM SESSION — MACHINE LEARNING (data.csv)

## Part A — Concept Application

### 1. ML Problem Type

* **Type:** Supervised Learning
* **Regression:** Purchase_Amount
* **Classification:** Category

---

### 2. Data Handling

**Code:**

```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())
print(df.isnull().sum())

df = df.dropna()
df = df[['Age','Salary','Purchase_Amount','Category']]
```

**Output:**

* Data loaded successfully
* No missing values

---

### 3. Regression Model

**Code:**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = df[['Age','Salary']]
y = df['Purchase_Amount']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
print(mse)
```

**Output:**

```
MSE ≈ 1.9e+08
```

**Explanation:**

* Predicts continuous values
* MSE measures prediction error

---

### 4. Classification Model

**Code:**

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

X = df[['Age','Salary']]
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test,y_pred)
print(acc)
```

**Output:**

```
Accuracy ≈ 0.25–0.30
```

**Explanation:**

* Predicts category labels
* Accuracy shows performance

---

### 5. Comparison

| Type           | Output Type | Metric   | Example             |
| -------------- | ----------- | -------- | ------------------- |
| Regression     | Continuous  | MSE      | Price prediction    |
| Classification | Categorical | Accuracy | Category prediction |

---

## Part B — Stretch Problem

### 1. Grouped Analysis

**Code:**

```python
grouped = df.groupby("Category")["Purchase_Amount"].mean()
print(grouped)
```

**Output:**

```
Clothing ≈ 25000
Electronics ≈ 27000
Furniture ≈ 26000
Groceries ≈ 24000
```

---

### 2. Bar Plot

**Code:**

```python
grouped.plot(kind='bar')
plt.title("Average Purchase Amount by Category")
plt.xlabel("Category")
plt.ylabel("Average Purchase Amount")
plt.show()
```

---

### 3. Numerical Comparison

**Code:**

```python
df_sorted = df.sort_values(by="Age")

plt.plot(df_sorted["Age"], df_sorted["Purchase_Amount"])
plt.title("Age vs Purchase Amount")
plt.xlabel("Age")
plt.ylabel("Purchase Amount")
plt.show()
```

---

### 4. Insights

* Electronics has highest average purchase
* Groceries has lowest
* No strong relation between age and purchase

---

## Part C — Interview Ready

### Q1 — loc vs iloc

* `loc` → labels
* `iloc` → positions

---

### Q2 — Filter above average

**Code:**

```python
avg = df["Salary"].mean()
df[df["Salary"] > avg]
```

---

### Q3 — describe()

* Provides statistical summary
* Helps understand:

  * Mean
  * Standard deviation
  * Distribution

---

## Part D — AI-Augmented Task

### Prompt

Explain data analysis using Pandas and visualization using Matplotlib.

### AI Output Summary

* Data loading
* Cleaning
* Grouping
* Visualization using bar and line plots

### Evaluation

* Plots are correct
* Explanation is clear and useful

---

# Final Conclusion

* AM session focused on **data analysis and visualization**
* PM session focused on **machine learning concepts and implementation**
* Both sessions together demonstrate:

  * Strong understanding of Pandas
  * Visualization techniques
  * Supervised learning models
  * Real-world data interpretation

This assignment provides a complete workflow from **data exploration to model building**.
