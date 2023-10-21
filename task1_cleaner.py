import pandas as pd
import random

from sklearn.preprocessing import StandardScaler

task1 = pd.read_csv('Solar_flare_RHESSI_2004_05.csv')

attributes = ['duration.s', 'total.counts', 'energy.kev.i', 'energy.kev.f', 'x.pos.asec', 'y.pos.asec', 'month', 'year']
working_data = task1[attributes]

# Batch 1
batch = ((working_data['month'].isin([1, 2, 3, 4])) & (working_data['year'] == 2004))
# Use the conditions to filter the DataFrame
df_1 = working_data[batch]

# Batch 2
batch = ((working_data['month'].isin([3, 4, 5, 6])) & (working_data['year'] == 2004))
df_2 = working_data[batch]

# Batch 3
batch = ((working_data['month'].isin([5, 6, 7, 8])) & (working_data['year'] == 2004))
df_3 = working_data[batch]

# Batch 4
batch = ((working_data['month'].isin([7, 8, 9, 10])) & (working_data['year'] == 2004))
df_4 = working_data[batch]

# Batch 5
batch = ((working_data['month'].isin([9, 10, 11, 12])) & (working_data['year'] == 2004))
df_5 = working_data[batch]

# Batch 6
batch = ((working_data['month'].isin([11, 12])) & (working_data['year'] == 2004)) | ((working_data['month'].isin([1, 2])) & (working_data['year'] == 2005))
df_6 = working_data[batch]

# Batch 7
batch = ((working_data['month'].isin([1, 2, 3, 4])) & (working_data['year'] == 2005))
df_7 = working_data[batch]

# Batch 8
batch = ((working_data['month'].isin([3, 4, 5, 6])) & (working_data['year'] == 2005))
df_8 = working_data[batch]

# Batch 9
batch = ((working_data['month'].isin([5, 6, 7, 8])) & (working_data['year'] == 2005))
df_9 = working_data[batch]

# Batch 10
batch = ((working_data['month'].isin([7, 8, 9, 10])) & (working_data['year'] == 2005))
df_10 = working_data[batch]

# Batch 11
batch = ((working_data['month'].isin([9, 10, 11, 12])) & (working_data['year'] == 2005))
df_11 = working_data[batch]

def fetch_intensity(x_value, y_value, radius, df):
    x_upper = x_value + radius
    x_lower = x_value - radius
    y_upper = y_value + radius
    y_lower = y_value - radius

    condition = ((x_lower <= df['x.pos.asec']) & (df['x.pos.asec'] <= x_upper)) & \
                ((y_lower <= df['y.pos.asec']) & (df['y.pos.asec'] <= y_upper))

    filtered_data = df[condition]
    count_addition = filtered_data['total.counts'].sum()

    new_row_data = {
        'x.pos.asec': x_value,
        'y.pos.asec': y_value,
        'total.counts': count_addition
    }

    return new_row_data, df[~condition]


def fetch_intensity_recursive(df):
    if df.empty:
        return pd.DataFrame(columns=['x.pos.asec', 'y.pos.asec', 'total.counts'])

    random_idx = random.randrange(len(df))
    x_value, y_value = df.iloc[random_idx, 4], df.iloc[random_idx, 5]

    new_row, remaining_df = fetch_intensity(x_value, y_value, 50, df)
    intensity_df = pd.DataFrame([new_row])

    return intensity_df._append(fetch_intensity_recursive(remaining_df), ignore_index=True)

final_intensity_list_batch_1 = fetch_intensity_recursive(df_1)
print(final_intensity_list_batch_1)
