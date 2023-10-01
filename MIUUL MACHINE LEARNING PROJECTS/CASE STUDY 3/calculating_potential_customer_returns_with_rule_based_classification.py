# Calculation of Potential Customer Revenue with Rule-Based Classification

## Business Problem

# A gaming company wants to create level-based new customer personas using certain characteristics of its customers and wants to predict how much new potential customers in these personas can potentially earn for the company.

# For example: They want to determine the average potential revenue of a 25-year-old male iOS user from Turkey.

## Dataset Story

# The dataset "Persona.csv" contains the prices of products sold by an international gaming company and some demographic information of users who purchased these products. The dataset consists of records generated for each transaction. This means that the table is not denormalized. In other words, a user with certain demographic characteristics may have made multiple purchases.

# Columns:
# - Price: The amount spent by the customer.
# - Source: The type of device the customer is using.
# - Sex: The gender of the customer.
# - Country: The country of the customer.
# - Age: The age of the customer.

## Before Application

"""
    PRICE   SOURCE   SEX COUNTRY  AGE
0     39  android  male     bra   17
1     39  android  male     bra   17
2     49  android  male     bra   17
3     29  android  male     tur   17
4     49  android  male     tur   17
"""

## After Application

"""
      customers_level_based        PRICE SEGMENT
0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
2  BRA_ANDROID_FEMALE_24_30   508.142857       A
3  BRA_ANDROID_FEMALE_31_40   233.166667       C
4  BRA_ANDROID_FEMALE_41_66   236.666667       C
"""


#############################################

# PROJECT TASKS
#############################################

#############################################

# TASK 1: Answer the following questions.
#############################################

# Question 1: Read the persona.csv file and display general information about the dataset.
import pandas as pd
df = pd.read_csv("datasets/persona.csv")
df.info()
df.head()
df.tail()
df.describe().T
df.shape
df.columns
df.isnull().any()

# Question 2: How many unique SOURCE values are there? What are their frequencies?

df["SOURCE"].unique()
df["SOURCE"].value_counts()

# Question 3: How many unique PRICE values are there?

df["PRICE"].nunique()

# Question 4: How many sales have been made for each PRICE?

df["PRICE"].value_counts()

# Question 5: How many sales have been made from each country?

df["COUNTRY"].value_counts()

# Question 6: How much revenue has been earned from each country?

df.groupby("COUNTRY").agg({"PRICE": "sum"})

# Question 7: What are the sales counts for each SOURCE type?

df.groupby("SOURCE").agg({"PRICE": "count"})  # df.SOURCE.value_counts()

# Question 8: What are the average PRICE values for each country?

df.groupby("COUNTRY").agg({"PRICE": "mean"})

# Question 9: What are the average PRICE values for each SOURCE?

df.groupby("SOURCE").agg({"PRICE": "mean"})

# Question 10: What are the average PRICE values for each COUNTRY-SOURCE combination?

df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# TASK 2: What are the average earnings by grouping data based on COUNTRY, SOURCE, SEX, and AGE?

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

# TASK 3: Sort the output based on PRICE.
# Apply the sort_values method on PRICE in descending order to the output from the previous question to visualize it better.
# Save the output as agg_df.

agg_df.sort_values(by="PRICE", ascending=False)

# TASK 4: Convert the index names to variable names.
# Convert all variables except for PRICE in the output of the third question to variable names.

agg_df = agg_df.reset_index()

# TASK 5: Convert the AGE variable into a categorical variable and add it to agg_df.
# Convert the numerical variable Age into a categorical variable.
# Create the intervals in a way that you believe would be meaningful.
# For example: '0_18', '19_23', '24_30', '31_40', '41_70'

my_binds = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
my_labels = ["0_18", "19_23", "24_30", "31_40", "41_" + str(agg_df["AGE"].max())]
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=my_binds, labels=my_labels)

# TASK 6: Define new level-based customer personas.
# Define new level-based customer personas and add them as a variable to the dataset.
# Variable name for the new addition: customers_level_based
# You need to create the customers_level_based variable by combining the observations obtained in the previous question.
# Attention! After creating the values for customers_level_based using List comprehension, it is necessary to make them unique.
# For example, there may be multiple occurrences of the following expression: USA_ANDROID_MALE_0_18.
# It is required to group them using groupby and calculate their average prices.


agg_df["CUSTOMERS_LEVEL_BASED"] = ["_".join(i).upper() for i in agg_df.drop(["AGE", "PRICE"], axis=1).values]
agg_df = agg_df[["CUSTOMERS_LEVEL_BASED", "PRICE"]]
agg_df = agg_df.groupby(["CUSTOMERS_LEVEL_BASED"]).agg({"PRICE": "mean"}).reset_index()


# TASK 7: Segment new customers (personas).
# Segment the new customers (Example: USA_ANDROID_MALE_0_18) according to PRICE into 4 segments.
# Add the segments as variables with the name SEGMENT to agg_df. Describe the segments (Group by segments and calculate the mean, max, and sum of price)

agg_df["SEGMENT"] = pd.qcut(agg_df.PRICE, q=4, labels=["D", "C", "B", "A"])
agg_df.head(5)
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]}).reset_index()

# TASK 8: Classify the incoming new customers and predict how much revenue they can generate.
# Which segment does a 33-year-old Turkish woman using ANDROID belong to, and what is the expected average revenue she would generate?
# Similarly, which segment does a 35-year-old French woman using IOS belong to, and what is the expected average revenue she would generate?

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user]

new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user2]