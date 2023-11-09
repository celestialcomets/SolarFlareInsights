import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
import random
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

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

    return intensity_df.append(fetch_intensity_recursive(remaining_df), ignore_index=True)

#_________Displaying Instensity List for Method 1_________
# Takes a final_intensity_list_batch# to display the data
def displayIntensityMethod1(intensity_data_frame, batch_num):
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

    title = f"Intensity Map Method 1 Batch {batch_num}"
    plt.title(title)
    plt.show()

# _________METHOD 2_________
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

#_________Displaying Instensity List for Method 2_________
# Made this second method to not re-normalize the data for display also 
#   second method returns DF with 'intensity' column not 'total.count'
# Takes a final_intensity_list_batch# to display the data
def displayIntensityMethod2(intensity_data_frame, batch_num):
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

    title = f"Intensity Map Method 2 Batch {batch_num}"
    plt.title(title)
    plt.show()

# _________DATA LOADING + SEPARATION_________
task1 = pd.read_csv('Solar_flare_RHESSI_2004_05.csv')
attributes_1 = ['duration.s', 'total.counts', 'energy.kev', 'x.pos.asec', 'y.pos.asec', 'month', 'year']
attributes_2 = ['duration.s', 'total.counts', 'energy.kev.i', 'energy.kev.f', 'x.pos.asec', 'y.pos.asec', 'month', 'year']
working_data_1 = task1[attributes_1]
working_data_2 = task1[attributes_2]

 
# Batch 1
batch = ((working_data_1['month'].isin([1, 2, 3, 4])) & (working_data_1['year'] == 2004))
# Use the conditions to filter the DataFrame
df_m1_b1 = working_data_1[batch]

batch = ((working_data_2['month'].isin([1, 2, 3, 4])) & (working_data_2['year'] == 2004))
df_m2_b1 = working_data_2[batch]

# Batch 2
batch = ((working_data_1['month'].isin([3, 4, 5, 6])) & (working_data_1['year'] == 2004))
df_m1_b2 = working_data_1[batch]

batch = ((working_data_2['month'].isin([3, 4, 5, 6])) & (working_data_2['year'] == 2004))
df_m2_b2 = working_data_2[batch]

# Batch 3
batch = ((working_data_1['month'].isin([5, 6, 7, 8])) & (working_data_1['year'] == 2004))
df_m1_b3 = working_data_1[batch]

batch = ((working_data_2['month'].isin([5, 6, 7, 8])) & (working_data_2['year'] == 2004))
df_m2_b3 = working_data_2[batch]

# Batch 4
batch = ((working_data_1['month'].isin([7, 8, 9, 10])) & (working_data_1['year'] == 2004))
df_m1_b4 = working_data_1[batch]

batch = ((working_data_2['month'].isin([7, 8, 9, 10])) & (working_data_2['year'] == 2004))
df_m2_b4 = working_data_2[batch]

# Batch 5
batch = ((working_data_1['month'].isin([9, 10, 11, 12])) & (working_data_1['year'] == 2004))
df_m1_b5 = working_data_1[batch]

batch = ((working_data_2['month'].isin([9, 10, 11, 12])) & (working_data_2['year'] == 2004))
df_m2_b5 = working_data_2[batch]

# Batch 6
batch = ((working_data_1['month'].isin([11, 12])) & (working_data_1['year'] == 2004)) | ((working_data_1['month'].isin([1, 2])) & (working_data_1['year'] == 2005))
df_m1_b6 = working_data_1[batch]

batch = ((working_data_2['month'].isin([11, 12])) & (working_data_2['year'] == 2004)) | ((working_data_2['month'].isin([1, 2])) & (working_data_2['year'] == 2005))
df_m2_b6 = working_data_2[batch]

# Batch 7
batch = ((working_data_1['month'].isin([1, 2, 3, 4])) & (working_data_1['year'] == 2005))
df_m1_b7 = working_data_1[batch]

batch = ((working_data_2['month'].isin([1, 2, 3, 4])) & (working_data_2['year'] == 2005))
df_m2_b7 = working_data_2[batch]

# Batch 8
batch = ((working_data_1['month'].isin([3, 4, 5, 6])) & (working_data_1['year'] == 2005))
df_m1_b8 = working_data_1[batch]

batch = ((working_data_2['month'].isin([3, 4, 5, 6])) & (working_data_2['year'] == 2005))
df_m2_b8 = working_data_2[batch]


# Batch 9
batch = ((working_data_1['month'].isin([5, 6, 7, 8])) & (working_data_1['year'] == 2005))
df_m1_b9 = working_data_1[batch]

batch = ((working_data_2['month'].isin([5, 6, 7, 8])) & (working_data_2['year'] == 2005))
df_m2_b9 = working_data_2[batch]

# Batch 10
batch = ((working_data_1['month'].isin([7, 8, 9, 10])) & (working_data_1['year'] == 2005))
df_m1_b10 = working_data_1[batch]

batch = ((working_data_2['month'].isin([7, 8, 9, 10])) & (working_data_2['year'] == 2005))
df_m2_b10 = working_data_2[batch]

# Batch 11
batch = ((working_data_1['month'].isin([9, 10, 11, 12])) & (working_data_1['year'] == 2005))
df_m1_b11 = working_data_1[batch]

batch = ((working_data_2['month'].isin([9, 10, 11, 12])) & (working_data_2['year'] == 2005))
df_m2_b11 = working_data_2[batch]

# _________BATCHES GO THROUGH METHOD 1_________
for i in range(1, 12, 1):
    # Define variable names
    batch_variable_name = f"final_intensity_list_batch_{i}_method_1"
    df_variable_name = f"df_m1_b{i}"
    
    # Get the current dataframe using its variable name
    current_df = globals()[df_variable_name]
    
    # Call fetch_intensity_recursive method and assign the result to the variable with dynamic name
    globals()[batch_variable_name] = fetch_intensity_recursive(current_df)
    
    # Print the result
    #print(globals()[batch_variable_name])

#print("final intensity list batch 5 method 1")
#print(final_intensity_list_batch_5_method_1.head())

# _________BATCHES GO THROUGH METHOD 2_________
for i in range(1, 12, 1):
    # Define variable names
    batch_variable_name = f"final_intensity_list_batch_{i}_method_2"
    df_variable_name = f"df_m2_b{i}"
    
    # Get the current dataframe using its variable name
    current_df = globals()[df_variable_name]

    current_df['energy.kev.mid'] = (current_df['energy.kev.i'] + current_df['energy.kev.f']) / 2
    scaler = StandardScaler()
    current_df[['energy.kev.mid', 'duration.s']] = scaler.fit_transform(current_df[['energy.kev.mid', 'duration.s']])
    
    # Call fetch_intensity_recursive method and assign the result to the variable with dynamic name
    globals()[batch_variable_name] = fetch_intensity_recursive_2(current_df)
    
    # Print the result
    #print(globals()[batch_variable_name])

#print("final intensity list batch 5 method 2")
#print(final_intensity_list_batch_5_method_2.head())

# intensity maps for months 1+2+3+4 using Method 1 and Method 2
#displayIntensityMethod1(final_intensity_list_batch_1_method_1, 1)
#displayIntensityMethod2(final_intensity_list_batch_1_method_2, 1)

# intensity maps for months 21+22+23+24 using Method 1 and Method 2
#displayIntensityMethod1(final_intensity_list_batch_11_method_1, 11)
#displayIntensityMethod2(final_intensity_list_batch_11_method_2, 11)

data = final_intensity_list_batch_1_method_1.to_numpy()
max_x = max(data[:,1])
min_x = min(data[:,1])
max_y = max(data[:,0])
min_y = min(data[:,0])
range_values = [[min_x, max_x],[min_y, max_y]]
grid_size = 25
hotspots, xbins, ybins = np.histogram2d(data[:,1], data[:,0], bins = grid_size, range=range_values)
plt.matshow(hotspots, extent=np.ravel([min_x, max_x, min_y, max_y]))
plt.colorbar()

hotspots = np.where(hotspots < 1.5, 0, hotspots)
plt.matshow(hotspots, extent=np.ravel([min_x, max_x, min_y, max_y]))
plt.colorbar()
plt.show()




data = final_intensity_list_batch_1_method_1.to_numpy()

# Define the range values for the bins
max_x = max(data[:, 1])
min_x = min(data[:, 1])
max_y = max(data[:, 0])
min_y = min(data[:, 0])
range_values = [[min_x, max_x], [min_y, max_y]]
grid_size = 25

# Calculate the histogram
hist, xedges, yedges = np.histogram2d(data[:, 1], data[:, 0], bins=grid_size, range=range_values)

# Define lower thresholds for the high and medium-high intensity spots
d1 = 1  # Lower threshold for high intensity
d2 = 0.5  # Lower threshold for medium-high intensity

# Create masks based on the adjusted thresholds
high_intensity_mask = np.where(hist > d1, 1, 0)
medium_high_intensity_mask = np.where((hist > d2) & (hist <= d1), 0.5, 0)  # Using 0.5 as a representative value for medium-high intensity

# Plot high-intensity and medium-high intensity spots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot high-intensity spots
cmap = ListedColormap(['white', 'red'])
im1 = axes[0].matshow(high_intensity_mask, extent=np.ravel([min_x, max_x, min_y, max_y]), cmap=cmap)
legend_elements1 = [Patch(facecolor='red', edgecolor='black', label='Intensity > 1')]
axes[0].legend(handles=legend_elements1, loc='upper right')
axes[0].set_title("High Intensity Hotspots")

# Plot medium-high intensity spots
cmap = ListedColormap(['white', 'orange'])
im2 = axes[1].matshow(medium_high_intensity_mask, extent=np.ravel([min_x, max_x, min_y, max_y]), cmap=cmap)
legend_elements2 = [Patch(facecolor='orange', edgecolor='black', label='0.5 < Intensity < 1')]
axes[1].legend(handles=legend_elements2, loc='upper right')
axes[1].set_title("Medium High Intensity Hotspots")

plt.tight_layout()
plt.show()

def plot_intensity(data, grid_size=25, d1=1, d2=0.5):
    # Define the range values for the bins
    max_x = max(data[:, 1])
    min_x = min(data[:, 1])
    max_y = max(data[:, 0])
    min_y = min(data[:, 0])
    range_values = [[min_x, max_x], [min_y, max_y]]

    # Calculate the histogram
    hist, xedges, yedges = np.histogram2d(data[:, 1], data[:, 0], bins=grid_size, range=range_values)

    # Create masks based on the adjusted thresholds
    high_intensity_mask = np.where(hist > d1, 1, 0)
    medium_high_intensity_mask = np.where((hist > d2) & (hist <= d1), 0.5, 0)

    # Plot high-intensity and medium-high intensity spots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot high-intensity spots
    cmap = ListedColormap(['white', 'red'])
    im1 = axes[0].matshow(high_intensity_mask, extent=np.ravel([min_x, max_x, min_y, max_y]), cmap=cmap)
    legend_elements1 = [Patch(facecolor='red', edgecolor='black', label='Intensity > 1')]
    axes[0].legend(handles=legend_elements1, loc='upper right')
    axes[0].set_title("High Intensity Hotspots")

    # Plot medium-high intensity spots
    cmap = ListedColormap(['white', 'orange'])
    im2 = axes[1].matshow(medium_high_intensity_mask, extent=np.ravel([min_x, max_x, min_y, max_y]), cmap=cmap)
    legend_elements2 = [Patch(facecolor='orange', edgecolor='black', label='0.5 < Intensity < 1')]
    axes[1].legend(handles=legend_elements2, loc='upper right')
    axes[1].set_title("Medium High Intensity Hotspots")

    plt.tight_layout()
    plt.show()

data = final_intensity_list_batch_1_method_1.to_numpy()

plot_intensity(data)