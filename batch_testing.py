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
# Use the conditions to filter t7:he DataFrame
df_1 = working_data[batch]
# Batch 2
batch2 = ((working_data['month'].isin([3, 4, 5, 6])) & (working_data['year'] == 2004))
df_2 = working_data[batch2]
#print(df_2)

# Batch 3
batch = ((working_data['month'].isin([5, 6, 7, 8])) & (working_data['year'] == 2004))

# Use the conditions to filter the DataFrame
df_3 = working_data[batch]

#print(df_3)

# Batch 4
batch = ((working_data['month'].isin([7, 8, 9, 10])) & (working_data['year'] == 2004))

# Use the conditions to filter the DataFrame
df_4 = working_data[batch]

#print(df_4)

# Batch 5
batch = ((working_data['month'].isin([9, 10, 11, 12])) & (working_data['year'] == 2004))

# Use the conditions to filter the DataFrame
df_5 = working_data[batch]

#print(df_5)

# Batch 6
batch = ((working_data['month'].isin([11, 12])) & (working_data['year'] == 2004)) | ((working_data['month'].isin([1, 2])) & (working_data['year'] == 2005))

# Use the conditions to filter the DataFrame
df_6 = working_data[batch]

#print(df_6)

# Batch 7
batch = ((working_data['month'].isin([1, 2, 3, 4])) & (working_data['year'] == 2005))

# Use the conditions to filter the DataFrame
df_7 = working_data[batch]

#print(df_7)

# Batch 8
batch = ((working_data['month'].isin([3, 4, 5, 6])) & (working_data['year'] == 2005))

# Use the conditions to filter the DataFrame
df_8 = working_data[batch]

#print(df_8)

# Batch 9
batch = ((working_data['month'].isin([5, 6, 7, 8])) & (working_data['year'] == 2005))

# Use the conditions to filter the DataFrame
df_9 = working_data[batch]

#print(df_9)

# Batch 10
batch = ((working_data['month'].isin([7, 8, 9, 10])) & (working_data['year'] == 2005))

# Use the conditions to filter the DataFrame
df_10 = working_data[batch]

#print(df_10)

# Batch 11
batch = ((working_data['month'].isin([9, 10, 11, 12])) & (working_data['year'] == 2005))

# Use the conditions to filter the DataFrame
df_11 = working_data[batch]

#print(df_11)

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

# _________BATCHES GO THROUGH METHOD 1_________
for i in range(1, 12, 1):
    # Define variable names
    batch_variable_name = f"final_intensity_list_batch_{i}_method_1"
    df_variable_name = f"df_{i}"
    
    # Get the current dataframe using its variable name
    current_df = globals()[df_variable_name]
    
    # Call fetch_intensity_recursive method and assign the result to the variable with dynamic name
    globals()[batch_variable_name] = fetch_intensity_recursive(current_df)
    
    # Print the result
    print(globals()[batch_variable_name])


# _________METHOD 2_________
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

x = final_intensity_list_batch_1_method_1['x.pos.asec'].values
y = final_intensity_list_batch_1_method_1['y.pos.asec'].values
total_counts = final_intensity_list_batch_1_method_1['total.counts'].values
low_threshold = 25000000
high_threshold = 50000000
xbins = 60
ybins = 60

# Create a custom colormap with red for high values and blue for low values
colors = np.where(total_counts > high_threshold, 'red', 'blue')

# Add a circle
x_center = 0
y_center = 0
radius = 1000

# Create a figure for the scatter plot and circle
fig, ax = plt.subplots()
ax.scatter(x, y, c=colors, marker='o', cmap='bwr')
circle = plt.Circle((x_center, y_center), radius, color='white', fill=False)
ax.add_artist(circle)

# Create a 2D histogram using np.histogram2d
hist, xedges, yedges = np.histogram2d(x, y, bins=(xbins, ybins))

# Create a mask to fill the area outside the circle
x_mesh, y_mesh = np.meshgrid(xedges, yedges)
distance_from_center = np.sqrt((x_mesh - x_center)**2 + (y_mesh - y_center)**2)
mask = distance_from_center > radius

# Ensure the mask dimensions match the histogram dimensions
mask = mask[:-1, :-1]  # Adjust the mask shape to match the histogram bins

# Mask the area outside the circle in the histogram
masked_hist = np.ma.masked_where(mask, hist)

# Plot the masked histogram
plt.figure()
plt.imshow(masked_hist, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='plasma')
plt.colorbar()
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('2D Histogram for Hotspot Discovery')

plt.show()