import pandas as pd

df = pd.read_csv("/app/sample_data.csv")
print(df.info())

print(df.describe())


print(df['label'].value_counts())

nan_values = df.isna().any().any()

unique_labels = df['label'].unique()
print(unique_labels)
print(df.shape[0])

if nan_values:
    print("There are NaN values in the DataFrame.")
else:
    print("There are no NaN values in the DataFrame.")