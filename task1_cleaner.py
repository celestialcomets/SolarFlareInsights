import pandas as pd
import random

task1 = pd.read_csv('Solar_flare_RHESSI_2004_05.csv')

attributes = ['duration.s', 'total.counts', 'energy.kev', 'x.pos.asec', 'y.pos.asec', 'month', 'year']
working_data = task1[attributes]

# Batch 1
batch = ((working_data['month'].isin([1, 2, 3, 4])) & (working_data['year'] == 2004))

# Use the conditions to filter the DataFrame
df_1 = working_data[batch]


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
    x_value, y_value = df.iloc[random_idx, 3], df.iloc[random_idx, 4]

    new_row, remaining_df = fetch_intensity(x_value, y_value, 50, df)
    intensity_df = pd.DataFrame([new_row])

    return intensity_df.append(fetch_intensity_recursive(remaining_df), ignore_index=True)


final_intensity_list_batch_1 = fetch_intensity_recursive(df_1)

print(final_intensity_list_batch_1)

