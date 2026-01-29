# 0️⃣ Import required libraries
import pandas as pd
import numpy as np

# --------------------------------------------
# 1️⃣ Load the dataset
# --------------------------------------------
df = pd.read_csv(r"C:\Users\admin\Desktop\Hunar-Intern-Data-Sci\dataset\food_coded.csv")

# --------------------------------------------
# 2️⃣ First glance at the data
# --------------------------------------------
print(df.head())
print(df.shape)
print(df.info())

# --------------------------------------------
# 3️⃣ Remove duplicate columns
# --------------------------------------------
df = df.loc[:, ~df.columns.duplicated()]

# --------------------------------------------
# 4️⃣ Convert fake NA values to real NaN
# --------------------------------------------
fake_na = [
    "NA", "N/A", "na", "n/a",
    "not sure", "Not sure",
    "none", "None",
    "zero", "Zero", ""
]

df = df.replace(fake_na, np.nan)

# --------------------------------------------
# 5️⃣ Handle Numeric Continuous Columns
#     (weight, GPA → median)
# --------------------------------------------
numeric_continuous = ['weight', 'GPA']

for col in numeric_continuous:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# --------------------------------------------
# 6️⃣ Numeric categorical columns → Mode
# --------------------------------------------
numeric_categorical = [
    'Gender', 'exercise', 'breakfast', 'coffee',
    'ethnic_food', 'greek_food', 'italian_food',
    'indian_food', 'thai_food', 'vitamins'
]

for col in numeric_categorical:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# --------------------------------------------
# 7️⃣ Text categorical columns → Mode
# --------------------------------------------
text_categorical = [
    'fav_cuisine',
    'father_profession',
    'mother_profession',
    'diet_current',
    'eating_changes'
]

for col in text_categorical:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# --------------------------------------------
# 8️⃣ Free-text columns → "Unknown"
# --------------------------------------------
free_text_columns = [
    'healthy_meal',
    'ideal_diet',
    'meals_dinner_friend',
    'type_sports'
]

for col in free_text_columns:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# --------------------------------------------
# 9️⃣ Remaining numeric columns → Median (bulk)
# --------------------------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# --------------------------------------------
# 🔟 Final validation
# --------------------------------------------
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFinal DataFrame info:")
print(df.info())

# --------------------------------------------
# 1️⃣1️⃣ Save cleaned dataset
# --------------------------------------------
df.to_csv(
    r"C:\Users\admin\Desktop\Hunar-Intern-Data-Sci\dataset\cleaned_food_coded.csv",
    index=False
)

print("\n✅ Data cleaning completed successfully!")
