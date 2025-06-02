import pandas as pd

# Step 1: Load CSV file
df = pd.read_csv('datascience_salaries.csv')  
# Step 2: Min-Max normalize the 'salary' column
min_salary = df['salary'].min()
max_salary = df['salary'].max()

df['normalized_salary'] = (df['salary'] - min_salary) / (max_salary - min_salary)


# Display the result
print(df[['location', 'salary', 'normalized_salary']])
