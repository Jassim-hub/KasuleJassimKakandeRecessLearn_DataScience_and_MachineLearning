import pandas as pd

# 1.	Write a Pandas program to compare the elements of two Pandas series.
series1 = pd.Series([4, 65, 436, 3, 9])
series2 = pd.Series([7, 0, 3, 897, 9])
print("Series1:\n", series1)
print("Series2:\n", series2)
print("\nElement-wise comparison (equal):")
print(series1 == series2)
print("\nElement-wise comparison (greater than):")
print(series1 > series2)
print("\nElement-wise comparison (less than):")
print(series1 < series2)
# 2.	Write a Pandas program to add, subtract, multiply and divide two Pandas series.
# Sample Series: [2,4,6,8,14], [1,3,5,7,9]
series3 = pd.Series([2, 4, 6, 8, 14])
series4 = pd.Series([1, 3, 5, 7, 9])
print("\nSeries3:", series3.tolist())
print("Series4:", series4.tolist())
print("\nAddition:")
print(series3 + series4)
print("\nSubtraction:")
print(series3 - series4)
print("\nMultiplication:")
print(series3 * series4)
print("\nDivision:")
print(series3 / series4)
# 3.Write a Pandas program to convert a dictionary to a Pandas series.
# Sample dictionary: dictionary1 = {‘Josh’: 24, ‘Sam’: 36, ‘Peace’: 19, ‘Charles’: 65, ‘Tom’: 44}
dictionary1 = {"Josh": 24, "Sam": 36, "Peace": 19, "Charles": 65, "Tom": 44}
series5 = pd.Series(dictionary1)
print("\nSeries5:")
print(series5)
# 4.	Write a Pandas program to convert a given series to an array.
# Sample series: [‘Love’, 800, ‘Joy’, 789.9, ‘Peace’, True]
sample_series = pd.Series(["Love", 800, "Joy", 789.9, "Peace", True])
print("\nSample Series:")
print(sample_series)
print("\nConverted to array:")
print(sample_series.values)
# 5. Write a Pandas program to display the most frequent value in the given series and replace everything else as 'Other' in the series.
# Using the 'HomeTeamGoals' column in the AfricaCupofNationsMatches.csv dataset
df = pd.read_csv("EXERCISE 2/AfricaCupofNationsMatches.csv")
home_goals = df["HomeTeamGoals"]
most_freq = home_goals.mode()[0]
print("\nMost frequent value in 'HomeTeamGoals':", most_freq)
modified_series = home_goals.apply(lambda x: x if x == most_freq else "Other")
print("\nModified Series (most frequent value, others replaced with 'Other'):")
print(modified_series)
