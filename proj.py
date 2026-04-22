# ================================
# UNIT II: DATA MANIPULATION
# ================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, shapiro, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

# Load Dataset
df = pd.read_csv("calorie_dataset.csv")

df.columns = df.columns.str.strip()

print("Columns:", df.columns)
print("\nFirst 5 rows:")
print(df.head())

# -------------------------------
# DATA CLEANING
# -------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

# ✅ FIXED WARNING HERE
cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
for col in cat_cols:
    df[col] = df[col].astype('category')

print("\nCleaned Data:")
print(df.head())

# Detect target column
target_col = None
for col in df.columns:
    if 'calorie' in col.lower():
        target_col = col
        break

if target_col is None:
    raise Exception("Calories column not found")

print("\nTarget Column:", target_col)

# ================================
# UNIT III: DATA VISUALIZATION
# ================================

num_df = df.select_dtypes(include=np.number)

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(df[target_col], bins=30)
plt.title("Distribution of Calories")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(df[num_df.columns[0]], df[target_col])
plt.title(f"{target_col} vs {num_df.columns[0]} (Scatter Plot)")
plt.xlabel(num_df.columns[0])
plt.ylabel(target_col)
plt.show()

# Box Plot
plt.figure(figsize=(10,6))
sns.boxplot(data=num_df)
plt.title("Box Plot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

 #✅ CLEAN LINE CHART (ONLY LINE)
df_sorted = df.sort_values(by=num_df.columns[0])

plt.figure(figsize=(8,5))
plt.plot(df_sorted[num_df.columns[0]], df_sorted[target_col])
plt.title(f"{target_col} vs {num_df.columns[0]} (Line Chart)")
plt.xlabel(num_df.columns[0])
plt.ylabel(target_col)
plt.grid(True)
plt.show()

# Count Plot
if len(cat_cols) > 0:
    plt.figure(figsize=(6,4))
    sns.countplot(x=cat_cols[0], data=df)
    plt.title(f"{cat_cols[0]} Distribution")
    plt.xticks(rotation=45)
    plt.show()

# Pie Chart
if len(cat_cols) > 0:
    counts = df[cat_cols[0]].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.title(f"{cat_cols[0]} Pie Chart")
    plt.show()

# Bar Plot
if len(cat_cols) > 0:
    plt.figure(figsize=(8,5))
    sns.barplot(x=cat_cols[0], y=target_col, data=df)
    plt.title(f"Average {target_col} by {cat_cols[0]}")
    plt.xticks(rotation=45)
    plt.show()

# ================================
# UNIT IV: EDA
# ================================

print("\nSummary Statistics:")
print(df.describe())

Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR)))

print("\nOutliers Count:")
print(outliers.sum())

# ================================
# UNIT V: STATISTICAL ANALYSIS
# ================================

print("\nNormality Test:")
for col in num_df.columns:
    try:
        _, p = shapiro(df[col])
        print(f"{col}: p-value = {p}")
    except:
        pass

if len(num_df.columns) >= 2:
    cols = num_df.columns
    _, p = ttest_ind(df[cols[0]], df[cols[1]])
    print("\nT-Test p-value:", p)

if len(cat_cols) >= 2:
    cont = pd.crosstab(df[cat_cols[0]], df[cat_cols[1]])
    _, p, _, _ = chi2_contingency(cont)
    print("Chi-Square p-value:", p)

# ================================
# UNIT VI: MACHINE LEARNING
# ================================

features = num_df.drop(columns=[target_col], errors='ignore')
target = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nLinear Regression R2 Score:", r2_score(y_test, y_pred))

# Logistic Regression
y_class = (target > target.mean()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    features, y_class, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, y_pred))
