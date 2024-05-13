import pandas as pd
#import numpy
import matplotlib.pyplot as plt
import random


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

def extract_coordinates(input_df):
    # Select the columns you need
    selected_columns = ['x.pos.asec', 'y.pos.asec', 'total.counts']

    # Create a new DataFrame with just the x and y coordinates
    new_df = input_df[selected_columns]

    return new_df


# Call the function with your DataFrame
coordinates_df = extract_coordinates(df_1) # Send first batch to only get x,y,count
print(coordinates_df) # 
# Assuming coordinates_df is your DataFrame
size_of_x_column = coordinates_df.shape[0] # Got the number of columns in df
print(size_of_x_column)
# Assuming size_of_first_column is already defined
random_number = random.randint(0, size_of_x_column - 1) # Got random number from size
print(random_number)


# For other batches simply create new copy,intensity_list
copy_coordinates_df = coordinates_df # Copy of x,y,count data frame
print(copy_coordinates_df)

final_columns = ['x.pos.asec', 'y.pos.asec', 'total.counts']
intensity_list_batch_1 = pd.DataFrame(columns=final_columns) # Blank data frame

#________________________________________________________________________________________________________________________
# JUST USED FOR TESTING
new_row_data = {
    'x.pos.asec': 10.5,
    'y.pos.asec': 20.0,
    'total.counts': 50
}

# Append the new row to the DataFrame using the concat method
intensity_list_batch_1 = pd.concat([intensity_list_batch_1, pd.DataFrame([new_row_data])], ignore_index=True)
print(intensity_list_batch_1)
#________________________________________________________________________________________________________________________


# Checking final count data frame it empty at beginning
print(intensity_list_batch_1)
intenstity_list_rows,intenstity_list_columns = intensity_list_batch_1.shape
print(intenstity_list_rows)
print(intenstity_list_columns)





# we could make a recursive function that will send the size of copy_coordinates_df 
# the intenstity_list_batch_1  until the copy is empty
def randomNum(copy_data_frame):
    max=copy_data_frame.shape[0]
    random_integer = random.randint(0, (max-1))
    return random_integer

def fetchData(copy_of_data_frame,random_number,intensity_list):
    #while(copy_coordinates_df is not None):
    #_____
    random_coordinate_row = copy_of_data_frame.iloc[random_number] # Get a whole role from random
    #_____
    
    #print(random_coordinate_row)
    ##print("-" *30)


    x_value = random_coordinate_row.iloc[0] # X value of row
    ##print(f"X-value: {x_value}")
    y_value = random_coordinate_row.iloc[1] # Y value of row
    ##print(f"Y-value: {y_value}")

    vari_zero = random_coordinate_row.iloc[2] # Count
    ##print(f"Count-value: {vari_zero}")
    ##print("-" *60)

    radius = 50 
    #radius = 25
    x_upper = x_value + radius
    x_lower = x_value - radius
    y_upper = y_value + radius
    y_lower = y_value - radius
    #getCoordinatesInRadius(x_lower,x_upper,y_lower,y_upper, copy_coordinates_df, instensity_list)

    # find the coordinates in the radius we decide on
    # sum up the value of those counts
    # remove those coordinates from the copy data frame
    # add the initial coordinate to the intesity_list with sum of counts
    # repeat

    #condition = (copy_of_data_frame.iloc[:, 0] >= x_value) & (copy_of_data_frame.iloc[:, 0] <= x_upper)

    #condition = ((copy_of_data_frame.iloc[:, 0] >= x_value) & (copy_of_data_frame.iloc[:, 0] <= x_upper)) & ((copy_of_data_frame.iloc[:, 1] >= y_value) & (copy_of_data_frame.iloc[:, 1] <= y_upper)) 
    #condition_2 = ((copy_of_data_frame.iloc[:, 0] <= x_value) & (copy_of_data_frame.iloc[:, 0] >= x_lower)) & ((copy_of_data_frame.iloc[:, 1] <= y_value) & (copy_of_data_frame.iloc[:, 1] >= y_lower))
    #&((copy_of_data_frame.iloc[:, 0] <= x_value) & (copy_of_data_frame.iloc[:, 0] >= x_lower)) & ((copy_of_data_frame.iloc[:, 1] <= y_value) & (copy_of_data_frame.iloc[:, 1] >= y_lower))
    condition = ((x_lower <= copy_of_data_frame.iloc[:, 0]) & (copy_of_data_frame.iloc[:, 0] <= x_upper)) & ((y_lower <= copy_of_data_frame.iloc[:, 1]) & (copy_of_data_frame.iloc[:, 1] <= y_upper))

    filtered_data = copy_of_data_frame[condition]
    #filtered_data_2 = copy_of_data_frame[condition_2]


    ##print(filtered_data)

    count_addition = 0
    # Loop through the 'filtered_data' DataFrame to sum row 2
    for index, row in filtered_data.iterrows():
        count_addition += row.iloc[2]

    ##print("-" *45)
    ##print(f"The Cound addition is: {count_addition}")
    ##print("-" *45)

    new_row_data = {
    'x.pos.asec': x_value,
    'y.pos.asec': y_value,
    'total.counts': count_addition
    }
    # Add the new row to the empty DataFrame
    #intensity_list = intensity_list.append(new_row_data, ignore_index=True)
    #intensity_list.append(new_row_data, ignore_index=True)
    intensity_list = pd.concat([intensity_list, pd.DataFrame([new_row_data])], ignore_index=True)
    ##print("Intensity list:")
    ##print(intensity_list)
    ##print("-" *45)

    copy_of_data_frame = copy_of_data_frame[~condition].reset_index(drop=True)
    updated_size = copy_of_data_frame.shape[0]
    
    #new_random_num = randomNum(copy_of_data_frame)
    ##print(copy_of_data_frame)
    
    #while copy_of_data_frame is not None:
    while not copy_of_data_frame.empty:
        new_random_num = randomNum(copy_of_data_frame)
        fetchData(copy_of_data_frame,new_random_num,intensity_list)
    #print(updated_size)
    #print(copy_of_data_frame)     
    return intensity_list,copy_of_data_frame #intensity_list

#final_intensity_list_batch_1 = fetchData(copy_coordinates_df,random_number,intenstity_list_batch_1)

start_number = randomNum(copy_coordinates_df)
final_intensity_list_batch_1,final_copy_df = fetchData(copy_coordinates_df,start_number,intensity_list_batch_1)
print(final_intensity_list_batch_1)
print(final_copy_df)




print(randomNum(copy_coordinates_df))

print(copy_coordinates_df)
