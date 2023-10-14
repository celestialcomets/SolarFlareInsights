import csv
import pandas as pd
# create df to see what we're looking at (helps subdivide)
attributes = ['duration.s','total.counts','energy.kev','x.pos.asec','y.pos.asec','month','year']
#edit to have it match your directory (below)
#raw data that we received
task1 = pd.read_csv('/Users/jerickaledezma/Desktop/COSC Group Project/cosc3337-groupproj/Solar_flare_RHESSI_2004_05.csv', 
                    header=None, names=attributes, skiprows=[0])

#we're subdiving task 1 data -> total 17 batches
# this is makes cleaner and easier to read
#month 1 to month 4
s = {'duration.s': task1['duration.s'], 'total.counts': task1['total.counts'], 'x.pos.asec': 
     task1['x.pos.asec'], 'y.pos.asec': task1['y.pos.asec'],'month': task1['month'],'year': task1['year']}
df = pd.DataFrame(s)
batch_1 = df['month'] == 1 #df['month'] == 1,2,3,4 # and (df['year'] == 2004)
df_1 = df[batch_1]
print(df_1)

s = {'duration.s': task1['duration.s'], 'total.counts': task1['total.counts'], 'x.pos.asec': 
     task1['x.pos.asec'], 'y.pos.asec': task1['y.pos.asec']}

sub_1 = pd.DataFrame(s)
#print(sub_1)
#month 3 to month 6

