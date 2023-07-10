import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.spatial.distance import cdist
import numpy as np
import chardet
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re

class SpatiotemporalApp:
    def __init__(self):
        self.data = None
        self.root = tk.Tk()
        self.root.geometry("350x350")
        self.root.title("Spatiotemporal Interpolation GUI")

        import_button = tk.Button(self.root, text="Import Data", command=self.import_data)
        import_button.pack()

        interpolate_button = tk.Button(self.root, text="Perform Interpolation", command=self.perform_interpolation_wrapper)
        interpolate_button.pack()

        error_metrics_button = tk.Button(self.root, text="Compute Error Metrics", command=self.compute_error_metrics)
        error_metrics_button.pack()

        query_button = tk.Button(self.root, text="Query Data", command=self.query_data)
        query_button.pack()

    def run(self):
        self.root.mainloop()

    def import_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            self.data = pd.read_csv(file_path, header=0, delimiter='\t', 
                        names=['id', 'year', 'month', 'day', 'x', 'y', 'measurement'], 
                        dtype={'id': int, 'year': int, 'month': int, 'day': int, 
                               'x': float, 'y': float, 'measurement': float})
            self.measurement_list = self.data['measurement'].tolist()
            time_domain = tk.simpledialog.askstring("Time domain", "Enter time domain (day, month, or year):")
            if time_domain is not None:
                if time_domain == "day" or time_domain =="1":
                    self.data['time'] = pd.to_datetime(self.data[['year', 'month', 'day']])
                elif time_domain == "month" or time_domain =="2":
                    self.data['time'] = pd.to_datetime(self.data[['year', 'month']].assign(day=1))
                elif time_domain == "year" or time_domain =="3":
                    self.data['time'] = pd.to_datetime(self.data[['year']].assign(month=1, day=1))
                visualize_data(self.data)
            else:
                messagebox.showerror("Error", "Please enter a time domain.")

    def perform_interpolation_wrapper(self):
        if self.data is not None:
            perform_interpolation(self.data, r"C:\Users\17708\Documents\School\Spring 2023\GIS\Project\county_xy.txt") #blkgrp_xy.txt, keep r in front for reading special characters
        else:
            messagebox.showerror("Error", "Please import data before performing interpolation.")

    def compute_error_metrics(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            data = pd.read_csv(file_path, delimiter=',')
            trueData = pd.read_csv("pm25_2009_measured.txt", delimiter='\t')
            with open("county_xy.txt", 'rb') as f:
                result = chardet.detect(f.read())
            encoding = result['encoding']

            locations = pd.read_csv("county_xy.txt", delimiter='\t', encoding=encoding)
            id_column = data['id']
            output_data = pd.DataFrame({'id': id_column})
            pattern = re.compile(r'\d+\.\d+')
            for col in data.columns[2:]:
                if col.startswith("n"): #Column names are 'n1e1' for example, and must be split
                    if "e" in col:
                        num_neighbors, exponent = col.split("e")
                        num_neighbors = int(num_neighbors[1:])
                        exponent = float(exponent)
                    else:
                        num_neighbors = int(col[1:])
                        exponent = 1.0
                    measurements = data[col].apply(lambda x: pd.Series(pattern.findall(x))).astype(float)
                    measured_data = pd.concat([id_column, measurements], axis=1).dropna()
                    predictions = measured_data.set_index('id').filter(regex='^\d', axis=1)
                    predictions = predictions.to_numpy(dtype=np.float64) # Convert predictions to a NumPy array with a proper data type

                    dist_matrix = cdist(locations[['x', 'y']], trueData[['x', 'y']])  # Calculate the nearest true_value index for each row in the data DataFrame
                    nearest_indices = np.argmin(dist_matrix, axis=1)

                    filtered_trueData = trueData.iloc[nearest_indices].reset_index(drop=True) # Extract the true_values from the trueData DataFrame using the calculated indices
                    true_values = filtered_trueData.loc[measured_data.index]['pm25']
                    predictions_mean = np.mean(predictions, axis=1)
                    mae = mean_absolute_error(true_values, predictions_mean)
                    mse = mean_squared_error(true_values, predictions_mean)
                    rmse = np.sqrt(mse)
                    mare = np.mean(np.abs((true_values - predictions_mean) / true_values))
                    output_data[f"{col}_MAE"] = mae
                    output_data[f"{col}_MSE"] = mse
                    output_data[f"{col}_RMSE"] = rmse
                    output_data[f"{col}_MARE"] = mare
            output_file = filedialog.asksaveasfilename(filetypes=[("Text files", "*.csv")])
            if output_file:
                output_data.to_csv(output_file, index=False)
        else:
            messagebox.showerror("Error", "Please select a file to import.")

    def query_data(self): #SELECT id, x, y, time, measurement FROM data WHERE time = 'YYYY-MM-DD' AND measurement BETWEEN min AND max
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            column_dtypes = {0: int, 1: int, 2: int, 3: int, 4: float}
            interpolated_data = pd.read_csv(file_path, delimiter=',', header=0, dtype=column_dtypes, low_memory=False)
            interpolated_data.columns = ['id', 'year', 'month', 'day'] + [f"measurement_{i}" for i in range(1, interpolated_data.shape[1] - 3)]

            # Get user inputs
            date = simpledialog.askstring("Query date", "Enter the date in the format 'YYYY-MM-DD':")
            min_val = simpledialog.askfloat("Minimum value", "Enter the minimum PM2.5 concentration:")
            max_val = simpledialog.askfloat("Maximum value", "Enter the maximum PM2.5 concentration:")

            year, month, day = map(int, date.split('-'))
            # Filter the DataFrame based on the user inputs
            filtered_data = interpolated_data[(interpolated_data['year'] == year) & 
                                            (interpolated_data['month'] == month) & 
                                            (interpolated_data['day'] == day)]

            if not filtered_data.empty:
                # Process and display the filtered data
                filtered_data = filtered_data[(filtered_data.iloc[:, 4:] >= min_val) & (filtered_data.iloc[:, 4:] <= max_val)]
                filtered_data.dropna(axis=1, how='all', inplace=True) #Remove spare columns
                filtered_data.dropna(inplace=True) #Remove NaN values from query
                print(filtered_data)
            else:
                print("No data found for the specified date.")

def visualize_data(data):
    fig, ax = plt.subplots(figsize=(15, 15))
    us_boundary = gpd.read_file("st99_d00.shp")
    us_boundary.boundary.plot(ax=ax)

    #Animation Function: Tried a lot of various things to get up and running. Unfortunately the use of 64-bit data stalled this.
    def update(frame_number):
        ax.clear()
        us_boundary.boundary.plot(ax=ax)
        daily_data = data[data['time'] == unique_times[frame_number]]
        sc = ax.scatter(daily_data['x'], daily_data['y'], c=daily_data['measurement'],
                        cmap='coolwarm', marker='o', alpha=0.5)
        ax.set_title(f"PM2.5 on {unique_times[frame_number].strftime('%Y-%m-%d')}")
        fig.colorbar(sc, ax=ax, label='PM2.5 concentration')

    unique_times = data['time'].unique()
    ani = FuncAnimation(fig, update, frames=len(unique_times), interval=200, blit=False)
    plt.show()

def idw_interpolation(measured_data, locations, num_neighbors, exponent):
    dist_matrix = cdist(measured_data[['x', 'y']], locations[['x', 'y']])
    nearest_indices = np.argpartition(dist_matrix, num_neighbors, axis=0)[:num_neighbors]

    dist_array = dist_matrix[nearest_indices]
    weights = 1 / (dist_array ** exponent)
    weights[np.isinf(weights)] = 0
    
    if nearest_indices.any():
        values = measured_data['measurement'].iloc[nearest_indices.ravel()].values
    else:
        values = np.array([])

    if not nearest_indices.size:
        return np.array([])

    values = values.reshape(nearest_indices.shape)  # Reshape the values array to match weights

    interpolation_result = np.zeros(locations.shape[0])
    for i in range(locations.shape[0]):
        interpolation_result[i] = np.sum(weights[:, i] * values[:, i].reshape(-1, 1)) / np.sum(weights[:, i])

    return interpolation_result

def perform_interpolation(data, locations_file):
    locations = read_csv_with_encoding(locations_file)
    num_neighbors_list = 0
    exponents_list = 0
    num_neighbors_list = simpledialog.askfloat("Neighbor Count", "Select how many neighbors:")
    if num_neighbors_list == 0:
        num_neighbors_list = [3, 4, 5, 6, 7]
    exponents_list = simpledialog.askfloat("Exponent", "Select the exponent:")
    if exponents_list == 0:
        exponents_list = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    print("Performing Interpolation...")

    unique_times = data['time'].unique()
    output_data = pd.DataFrame({'id': locations['id']})
    output_data['original'] = ''

    for index, row in output_data.iterrows():
        county_id = row['id']
        original_measurements = data[data['id'] == county_id]['measurement'].astype(str).tolist()
        output_data.at[index, 'original'] = '  '.join(original_measurements)

    for num_neighbors in num_neighbors_list:
        for exponent in exponents_list:
            column_name = f"n{num_neighbors}e{exponent}"
            interpolated_values_all = []
            for time in unique_times:
                measured_data = data[data['time'] == time]
                interpolated_values = idw_interpolation(measured_data, locations, num_neighbors, exponent)
                interpolated_values_all.extend(interpolated_values)
            interpolated_values_all = np.array(interpolated_values_all)
            num_locations = len(locations)
            interpolated_values_all = interpolated_values_all.reshape(len(unique_times), num_locations)
            output_data[column_name] = list(interpolated_values_all.T)

    output_file = filedialog.asksaveasfilename(filetypes=[("Text files", "*.txt")])
    if output_file:
        output_data.to_csv(output_file, index=False)

def read_csv_with_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    
    df = pd.read_csv(file_path, encoding=result['encoding'], delimiter='\t')
    #print(df.columns)  # Keep this line for now to check the columns
    return df

def idw_interpolation_loocv(measured_data, locations, num_neighbors, exponent):
    errors = []
    for i in range(len(measured_data)):
        train_data = measured_data.drop(measured_data.index[i])
        test_data = measured_data.iloc[i]
        interpolated_value = idw_interpolation(train_data, pd.DataFrame([test_data[['x', 'y']]]), num_neighbors, exponent)
        error = test_data['measurement'] - interpolated_value[0]
        errors.append(error)
    mse = np.mean(np.array(errors) ** 2)
    return mse

if __name__ == "__main__":
    app = SpatiotemporalApp()
    app.run()


