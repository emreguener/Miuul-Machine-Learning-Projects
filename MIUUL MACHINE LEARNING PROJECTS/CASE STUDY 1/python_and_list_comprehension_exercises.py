# Convert all letters of the given string to uppercase. Replace commas and periods with spaces,
# and split the text into words.

text = "The goal is to turn data into information, and information into insight."

new_text = list()

for i in text:
    if i == ',' or i == '.':
        new_text.append(' ')
    else:
        new_text.append(i.upper())

new_text = ''.join(new_text)  # Join the list elements to obtain a string
word_list = new_text.split()  # Split the string by spaces to obtain a list of words
print(word_list)

#################################################################################################

# Apply the following steps to the given list:

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Find the number of elements in the given list.
print(len(lst))

# Retrieve elements at the 0th and 10th indices.
print(lst[0], lst[10])

# Create a new list containing ["D", "A", "T", "A"] from the given list.
# 1st way
new_lst = []
a = 0
for i in lst:
    if a < 4:
        new_lst.append(i)
    a += 1
print(new_lst)

# 2nd way
print(lst[0:4])

# Remove the element at the 8th index.
lst.pop(8)
print(lst)

# Add a new element.
lst.append("Q")
print(lst)

# Re-add the "N" element at the 8th index.
lst.insert(8, "N")
print(lst)

###########################################################################################

# Apply the following steps to the given dictionary:

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

# Access key and value.
dict.keys()
dict.values()

# Update the value "12" for the key "Daisy" to "13".
dict['Daisy'][1] = 13

# Add a new key-value pair with key "Ahmet" and value ["Turkey", 24].
dict['Ahmet'] = ["Turkey", 24]

# Remove "Antonio" from the dictionary.
dict.pop('Antonio')
print(dict)

##########################################################################################

# Write a function that takes a list as an argument, separates even and odd numbers in the list into two lists,
# and returns these lists.

l = [2, 13, 18, 93, 22]

def separate_numbers(list):
    even_list = []
    odd_list = []
    for number in list:
        if number % 2 == 0:
            even_list.append(number)
        else:
            odd_list.append(number)

    return even_list, odd_list

even_numbers, odd_numbers = separate_numbers(l)

print(even_numbers)
print(odd_numbers)

############################################################

# In the list provided below, the names of students who have achieved success in the engineering and medical faculties
# are listed. The first three students represent the ranking of the engineering faculty, while the last three students
# represent the ranking of the medical faculty. Using enumerate, print the student rankings by faculty.

students = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]
faculties = ["Engineering Faculty", "Medical Faculty"]

for index, student in enumerate(students, 1):
    if index < 4:
        print(faculties[0] + " Student " + str(index) + ": " + student)
    else:
        print(faculties[1] + " Student " + str(index - 3) + ": " + student)

###############################################################################

# The following three lists contain information about a course, including the course code, credits, and capacity.
# Using zip, print the course information.

course_codes = ["CMP1005", "PSY1001", "LAW1005", "SEN2204"]
credits = [3, 4, 2, 4]
capacities = [30, 75, 150, 25]

zipped = list(zip(credits, course_codes, capacities))

for credit, code, capacity in zipped:
    print(f"The course with code {code} has {credit} credits and a capacity of {capacity} students.")

############################################################################

# Two sets are provided below. You are asked to define a function that, if the first set includes the second set,
# prints the common elements; if not, prints the elements unique to the second set.

def set_operation(set1, set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))

set1 = set(["data", "python"])
set2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

set_operation(set1, set2)

##############################################################################################
##############################################################################################
# COMPREHENSION

# Using List Comprehension, convert the names of numeric variables in the car_crashes dataset to uppercase and add "NUM_" prefix.

import seaborn as sns
df = sns.load_dataset("car_crashes")
print(df.columns)
# Check if the column is not of type "Object" to determine if it is numeric.
print(["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns])

# Using List Comprehension, add "_FLAG" to the names of variables in the car_crashes dataset that do not contain "no" in their names.

print([col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns])

##################################################################################

# Using List Comprehension, select variable names in the car_crashes dataset that are NOT in the given list of original variable names,
# and create a new data frame with these variable names.

original_list = ["abbrev", "no_previous"]

print(df.columns)

new_cols = [col for col in df.columns if col not in original_list]

new_df = df[new_cols]

print(new_df.head())
