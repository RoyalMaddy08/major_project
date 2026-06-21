import pandas as pd
df = pd.read_csv("ev_battery_dataset_multiclass.csv")
print(df.columns)
data = df[['Voltage', 'Current', 'Temperature', 'State']].values
data_slice = data[200:700]
print(set(data_slice[:,3]))
