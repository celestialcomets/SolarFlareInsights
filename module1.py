import csv
import pandas as pd

task1 = pd.read_csv('Solar_flare_RHESSI_2004_05.csv')

attributes = ['duration.s','total.counts','energy.kev','x.pos.asec','y.pos.asec','month','year']

working_data = task1[attributes]

s = {'duration.s': working_data['duration.s'], 'total.counts': working_data['total.counts'], 'x.pos.asec':
     working_data['x.pos.asec'], 'y.pos.asec': working_data['y.pos.asec'],'month': working_data['month'],'year': working_data['year']}
df = pd.DataFrame(s)

batch_1 = (df['month'] == 1 or df['month'] == 2 or df['month'] == 3 or df['month'] == 4) # and (df['year'] == 2004)
df_1 = df[batch_1]

print(df_1)




