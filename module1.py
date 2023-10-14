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


