import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
import random

# _________DATA SEPARATION_________
task1 = pd.read_csv('Solar_flare_RHESSI_2004_05.csv')
attributes = ['duration.s', 'total.counts', 'energy.kev', 'x.pos.asec', 'y.pos.asec', 'month', 'year']
working_data = task1[attributes]
# Batch 1
batch = ((working_data['month'].isin([1, 2, 3, 4])) & (working_data['year'] == 2004))
# Use the conditions to filter the DataFrame
df_1 = working_data[batch]

# _________METHOD 1_________
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

    return intensity_df._append(fetch_intensity_recursive(remaining_df), ignore_index=True)

# _________BATCH 1 METHOD 1_________
final_intensity_list_batch_1_method_1 = fetch_intensity_recursive(df_1)
print(final_intensity_list_batch_1_method_1)


# _________METHOD 2_________
attributes = ['duration.s', 'total.counts', 'energy.kev.i', 'energy.kev.f', 'x.pos.asec', 'y.pos.asec', 'month', 'year']
working_data = task1[attributes]

# Batch 1
batch = ((working_data['month'].isin([1, 2, 3, 4])) & (working_data['year'] == 2004))
# Use the conditions to filter the DataFrame
df_1 = working_data[batch]
df_1['energy.kev.mid'] = (df_1['energy.kev.i'] + df_1['energy.kev.f']) / 2
scaler = StandardScaler()
df_1[['energy.kev.mid', 'duration.s']] = scaler.fit_transform(df_1[['energy.kev.mid', 'duration.s']])
print(df_1[['energy.kev.mid', 'duration.s']])

def fetch_intensity_2(x_value, y_value, radius, df):
    x_upper = x_value + radius
    x_lower = x_value - radius
    y_upper = y_value + radius
    y_lower = y_value - radius

    condition = ((x_lower <= df['x.pos.asec']) & (df['x.pos.asec'] <= x_upper)) & \
                ((y_lower <= df['y.pos.asec']) & (df['y.pos.asec'] <= y_upper))

    filtered_data = df[condition]
    intensity = (filtered_data['duration.s'] * filtered_data['energy.kev.mid']).sum()

    new_row_data = {
        'x.pos.asec': x_value,
        'y.pos.asec': y_value,
        'intensity': intensity
    }

    return new_row_data, df[~condition]

def fetch_intensity_recursive_2(df):
    intensity_list = []

    while not df.empty:
        random_idx = random.randrange(len(df))
        x_value, y_value = df.iloc[random_idx, 4], df.iloc[random_idx, 5]
        new_row, df = fetch_intensity_2(x_value, y_value, 50, df)
        intensity_list.append(new_row)

    return pd.DataFrame(intensity_list)

# _________BATCH 1 METHOD 2_________
final_intensity_list_batch_1_method_2 = fetch_intensity_recursive_2(df_1)
print(final_intensity_list_batch_1_method_2)


#_________Displaying Instensity List for Method 1_________
# Takes a final_intensity_list_batch# to display the data
def displayIntensityMethod1(intensity_data_frame):
    # Displaying the Intensity Map
    x_coordinates = intensity_data_frame['x.pos.asec'].values
    y_coordinates = intensity_data_frame['y.pos.asec'].values
    intensity_values = intensity_data_frame['total.counts'].values
    num_points = len(x_coordinates)  # Get the number of data points

    # Normalize intensity values to create a gradient colormap
    norm = Normalize(vmin=min(intensity_values), vmax=max(intensity_values))
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    
    # We can change s to make the circles smaller 
    scatter = ax.scatter(x_coordinates, y_coordinates, c=intensity_values, cmap='viridis', s=40, norm=norm)

    # Set the axis limits to create a circular plot
    max_radius = max(np.max(np.abs(x_coordinates)), np.max(np.abs(y_coordinates)))
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)

    # Add a circle for reference
    circle = plt.Circle((0, 0), max_radius, fill=False, color='r')
    ax.add_artist(circle)

    # Add lines through the circle
    ax.plot([-max_radius, max_radius], [0, 0], 'r--', lw=2)  # Horizontal line
    ax.plot([0, 0], [-max_radius, max_radius], 'r--', lw=2)  # Vertical line

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Intensity')

    plt.title("Intensity Map Method 1")
    plt.show()

displayIntensityMethod1(final_intensity_list_batch_1_method_1)


#_________Displaying Instensity List for Method 2_________
# Made this second method to not re-normalize the data for display also 
#   second method returns DF with 'intensity' column not 'total.count'
# Takes a final_intensity_list_batch# to display the data
def displayIntensityMethod2(intensity_data_frame):
    # Displaying the Intensity Map
    x_coordinates = intensity_data_frame['x.pos.asec'].values
    y_coordinates = intensity_data_frame['y.pos.asec'].values
    intensity_values = intensity_data_frame['intensity'].values  # Assuming these values are already normalized

    num_points = len(x_coordinates)  # Get the number of data points
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    
    # We can change s to make the circles smaller 
    scatter = ax.scatter(x_coordinates, y_coordinates, c=intensity_values, cmap='viridis', s=40)

    # Set the axis limits to create a circular plot
    max_radius = max(np.max(np.abs(x_coordinates)), np.max(np.abs(y_coordinates)))
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)

    # Add a circle for reference
    circle = plt.Circle((0, 0), max_radius, fill=False, color='r')
    ax.add_artist(circle)

    # Add lines through the circle
    ax.plot([-max_radius, max_radius], [0, 0], 'r--', lw=2)  # Horizontal line
    ax.plot([0, 0], [-max_radius, max_radius], 'r--', lw=2)  # Vertical line

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Intensity')

    plt.title("Intensity Map Method 2")
    plt.show()

displayIntensityMethod2(final_intensity_list_batch_1_method_2)
