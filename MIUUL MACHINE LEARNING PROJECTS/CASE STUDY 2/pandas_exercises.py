# Import the necessary libraries
import seaborn as sns
import pandas as pd

# Set display options for Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Task 1: Define the Titanic dataset from the Seaborn library.
df = sns.load_dataset("titanic")

# Task 2: Find the number of male and female passengers in the Titanic dataset.
male = df["sex"].value_counts()["male"]
female = df["sex"].value_counts()['female']

# Task 3: Find the number of unique values for each column.
df.nunique()

# Task 4: Find the number of unique values for the 'pclass' variable.
df["pclass"].nunique()

# Task 5: Find the number of unique values for the 'pclass' and 'parch' variables.
df[["pclass", "parch"]].nunique()

# Task 6: Check the data type of the 'embarked' variable, change it to 'category', and check again.
df["embarked"].dtype
df["embarked"].astype("category")

# Task 7: Show all information for passengers with 'embarked' value 'C'.
df[df["embarked"] == "C"]

# Task 8: Show all information for passengers with 'embarked' value other than 'S'.
df[df["embarked"] != "S"]

# Task 9: Show all information for passengers who are female and under 30 years old.
df[(df["age"] < 30) & (df["sex"] == "female")]

# Task 10: Show information for passengers with fare greater than 500 or age greater than 70.
df[(df["fare"] > 500) | (df["age"] > 70)]

# Task 11: Find the total count of missing values in each column.
df.isnull().sum()

# Task 12: Remove the 'who' variable from the dataframe.
df.drop("who", axis=1, inplace=True)

# Task 13: Fill missing values in the 'deck' variable with the mode (most frequent value).
df["deck"].fillna(df["deck"].mode()[0], inplace=True)

# Task 14: Fill missing values in the 'age' variable with the median.
df["age"].fillna(df["age"].median(), inplace=True)

# Task 15: Find sum, count, and mean values of 'survived' variable grouped by 'pclass' and 'sex'.
df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]})

# Task 16: Create a new variable 'age_flag' with 1 for ages under 30 and 0 for ages 30 and above.
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

# Task 17: Define the Tips dataset from the Seaborn library.
df = sns.load_dataset("tips")

# Task 18: Find the sum, min, max, and mean of 'total_bill' based on 'time' categories (Dinner, Lunch).
df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]})

# Task 19: Find the sum, min, max, and mean of 'total_bill' based on 'day' and 'time' categories.
df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

# Task 20: Find the sum, min, max, and mean of 'total_bill' and 'tip' for Lunch and female customers, grouped by 'day'.
df_female_lunch = df[(df["time"] == "Lunch") & (df["sex"] == "Female")]
df_female_lunch.groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                                     "tip": ["sum", "min", "max", "mean"]})

# Task 21: Calculate the mean total bill for orders with 'size' less than 3 and 'total_bill' greater than 10.
df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()

# Task 22: Create a new variable 'total_bill_tip_sum' that represents the sum of 'total_bill' and 'tip' for each customer.
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

# Task 23: Sort the dataframe by 'total_bill_tip_sum' in descending order and select the top 30 rows.
new_df = df.sort_values(by="total_bill_tip_sum", ascending=False).head(30)

