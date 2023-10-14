import csv
import pandas as pd

task1 = pd.read_csv('Solar_flare_RHESSI_2004_05.csv')

attributes = ['duration.s','total.counts','energy.kev','x.pos.asec','y.pos.asec','month','year']

working_data = task1[attributes]

s = {'duration.s': working_data['duration.s'], 'total.counts': working_data['total.counts'], 'x.pos.asec':
     working_data['x.pos.asec'], 'y.pos.asec': working_data['y.pos.asec'],'month': working_data['month'],'year': working_data['year']}
df = pd.DataFrame(s)

# Batch 1
batch = ((df['month'].isin([1, 2, 3, 4])) & (df['year'] == 2004))

# Use the conditions to filter the DataFrame
df_1 = df[batch]

print(df_1)

# Batch 2
batch = ((df['month'].isin([3, 4, 5, 6])) & (df['year'] == 2004))

# Use the conditions to filter the DataFrame
df_2 = df[batch]

print(df_2)

# Batch 3
batch = ((df['month'].isin([5, 6, 7, 8])) & (df['year'] == 2004))

# Use the conditions to filter the DataFrame
df_3 = df[batch]

print(df_3)

# Batch 4
batch = ((df['month'].isin([7, 8, 9, 10])) & (df['year'] == 2004))

# Use the conditions to filter the DataFrame
df_4 = df[batch]

print(df_4)

# Batch 5
batch = ((df['month'].isin([9, 10, 11, 12])) & (df['year'] == 2004))

# Use the conditions to filter the DataFrame
df_5 = df[batch]

print(df_5)

# Batch 6
batch = ((df['month'].isin([11, 12])) & (df['year'] == 2004)) | ((df['month'].isin([1, 2])) & (df['year'] == 2005))

# Use the conditions to filter the DataFrame
df_6 = df[batch]

print(df_6)

# Batch 7
batch = ((df['month'].isin([1, 2, 3, 4])) & (df['year'] == 2005))

# Use the conditions to filter the DataFrame
df_7 = df[batch]

print(df_7)

# Batch 8
batch = ((df['month'].isin([3, 4, 5, 6])) & (df['year'] == 2005))

# Use the conditions to filter the DataFrame
df_8 = df[batch]

print(df_8)

# Batch 9
batch = ((df['month'].isin([5, 6, 7, 8])) & (df['year'] == 2005))

# Use the conditions to filter the DataFrame
df_9 = df[batch]

print(df_9)

# Batch 10
batch = ((df['month'].isin([7, 8, 9, 10])) & (df['year'] == 2005))

# Use the conditions to filter the DataFrame
df_10 = df[batch]

print(df_10)

# Batch 11
batch = ((df['month'].isin([9, 10, 11, 12])) & (df['year'] == 2005))

# Use the conditions to filter the DataFrame
df_11 = df[batch]

print(df_11)
