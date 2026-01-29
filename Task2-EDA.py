# 📊 TASK 2: Exploratory Data Analysis (EDA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Better plot appearance
sns.set(style="whitegrid")


## 2️⃣ Load cleaned dataset

df = pd.read_csv(r"C:\Users\admin\Desktop\Hunar-Intern-Data-Sci\dataset\cleaned_food_coded.csv")



## 3️⃣ Basic EDA – First understanding

df.head()
df.shape
df.info()
df.describe()


## 4️⃣ Separate Numerical & Categorical Columns

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include='object').columns

## 5️⃣ Distribution of Key Numerical Variables

### 🎯 GPA Distribution

plt.figure(figsize=(6,4))
sns.histplot(df['GPA'], kde=True)
plt.title("GPA Distribution")
plt.xlabel("GPA")
plt.ylabel("Count")
plt.show()




### 🎯 Weight Distribution

plt.figure(figsize=(6,4))
sns.histplot(df['weight'], kde=True)
plt.title("Weight Distribution")
plt.xlabel("Weight")
plt.ylabel("Count")
plt.show()
## 6️⃣ Outlier Detection (Boxplots)

### Calories per Day
plt.figure(figsize=(6,4))
sns.boxplot(x=df['calories_day'])
plt.title("Calories Per Day (Outliers)")
plt.show()

### Income

plt.figure(figsize=(6,4))
sns.boxplot(x=df['income'])
plt.title("Income Distribution")
plt.show()

## 7️⃣ Categorical Variable Analysis

### Gender Distribution

plt.figure(figsize=(5,4))
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.show()

### Exercise Frequency

plt.figure(figsize=(6,4))
sns.countplot(x='exercise', data=df)
plt.title("Exercise Frequency")
plt.show()

## 8️⃣ Relationship Analysis (IMPORTANT)

### GPA vs Exercise

plt.figure(figsize=(6,4))
sns.boxplot(x='exercise', y='GPA', data=df)
plt.title("GPA vs Exercise Level")
plt.show()

### GPA vs Breakfast

plt.figure(figsize=(6,4))
sns.boxplot(x='breakfast', y='GPA', data=df)
plt.title("GPA vs Breakfast Habit")
plt.show()




## 9️⃣ Correlation Heatmap (Numerical Features)

plt.figure(figsize=(14,10))
corr = df[num_cols].corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Heatmap (Numerical Features)")
plt.show()


## 🔟 Comfort Food Analysis (Text Insight)

df['comfort_food'].value_counts().head(10)


df['comfort_food'].value_counts().head(10).plot(kind='bar')
plt.title("Top Comfort Foods")
plt.show()




## 1️⃣1️⃣ Lifestyle vs Health Perception

plt.figure(figsize=(6,4))
sns.boxplot(x='healthy_feeling', y='GPA', data=df)
plt.title("Healthy Feeling vs GPA")
plt.show()

## 1️⃣2️⃣ Save EDA-ready dataset (optional)
#df.to_csv("eda_ready_dataset.csv", index=False)

