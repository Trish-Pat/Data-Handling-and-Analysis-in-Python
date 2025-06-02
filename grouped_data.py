import pandas as pd

# Load dataset
df = pd.read_csv('datascience_salaries.csv') 

# Group by 'experience_level' and calculate average and median salary
grouped = df.groupby('experience_level')['salary'].agg(['mean', 'median']).reset_index()

# Rename columns for clarity
grouped.columns = ['Experience Level', 'Average Salary', 'Median Salary']

# Display the result
print(grouped)
