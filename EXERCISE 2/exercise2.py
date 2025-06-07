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
