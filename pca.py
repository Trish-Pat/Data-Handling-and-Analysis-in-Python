import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('datascience_salaries.csv') 

#Drop non-numeric columns (for PCA/t-SNE to work)
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Scale the data (important for PCA and t-SNE)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_df)
