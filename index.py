import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
students = pd.read_csv('students.csv')

# Display the first 5 rows of the DataFrame
print(students.head())

# Display basic statistics for numerical columns
statistics_summary = students.describe()
print(statistics_summary)

# Check for missing values
missing_values = students.isnull().sum()
print(missing_values)

#filling missing values with the count
students.fillna(students.count(), inplace=True)

# Create a new column 'Total'
students['Total'] = students.iloc[:, -3:].sum(axis=1)
print(students)

# Calculate Percentage and add a new column
students['Percentage'] = (students['Total'] / 300) * 100
print(students)

# Identify and display the student(s) with the highest percentage
highest_percentage_student = students[students['Percentage'] == students['Percentage'].max()]
print("highest_percentage_student", highest_percentage_student)

# Create a new DataFrame with students above the average percentage
average_percentage = students['Percentage'].mean()
above_average_students = students[students['Percentage'] > average_percentage]
print("above_average_students", above_average_students)

# Plot histogram of 'Total' marks
plt.hist(students['Total'])
plt.show()

# Plot bar of 'average marks' for subjects
average_marks_subjects = students[['Math_Score', 'English_Score', 'Science_Score']].mean()
plt.bar(average_marks_subjects.index, average_marks_subjects.values, color=['green', 'yellow', 'red'])
plt.xlabel('Subjects')
plt.ylabel('Average Marks')
plt.title('Average Marks in Each Subject')
plt.show()

plt.scatter(students['Total'], students['Percentage'])
plt.xlabel('Total Marks')
plt.ylabel('Percentage')
plt.title('Scatter Plot of Percentage vs Total Marks')
plt.show()