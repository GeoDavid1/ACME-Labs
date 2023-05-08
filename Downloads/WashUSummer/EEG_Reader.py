# Import necessary modules
import os
import datetime
from pathlib import Path
import numpy as np
import h5py

# from scipy import signal
import matplotlib.pyplot as plt
from pyedflib import highlevel

RAW_ROOT = "/Volumes/raw_data/2023/"
EDF_ROOT = "/Users/user/SynologyDrive/data/"


def make_paths(folder=RAW_ROOT):

    # Find the paths that start with ANTSC
    paths = os.listdir(folder)
    antsc_paths = [path for path in paths if path.startswith("ANTSC")]

    # Return these paths
    return antsc_paths


def time_difference(time1, time2):

    """
    This function returns the difference between two times in seconds.
    """

    # Get the difference in seconds
    diff = time2 - time1

    # Convert the difference in seconds to a human-readable format
    return (datetime.timedelta(seconds=diff)).total_seconds()


def iterate_through_files(path):

    files = os.listdir(path)
    return files


def makelist_of_data(file_path, time_run=0.020):

    """ " This function collects data from .h5 files in a directory and returns a concatenated list of EEG/EMG data.

    Args:
        file_path (str): The file path of the directory containing the .h5 files.
        time_run (float): The maximum amount of time allowed to convert the data

    Returns:
        concatenated_list (list): A concatenated list of EEG/EMG data.
        time_stamp_array (numpy.ndarray): An array of timestamps.
        time_accumulated_array (numpy.ndarray): An array of accumulated times.

    """

    # Initizalize the time accumulator
    time_accumulated = 0
    time_stamp_array = []
    time_accumulated_array = []
    eeg_emg_lists = []

    # Iterate over the files in the directory
    for filename in iterate_through_files(file_path):

        # Get the file path
        file = os.path.join(file_path, filename)

        # Open the file
        with h5py.File(file, "r") as f:

            # Get the file contents
            eeg_and_emg = f["data"][:]
            # row, col = np.shape(eeg_and_emg)
            time_array = f["t"][:]

            # Compute elapsed time in collecting data
            start_time = time_array[0]
            end_time = time_array[-1]
            diff_time = time_difference(start_time, end_time) / (
                60 * 60
            )  # Convert seconds into hours

            # Find number of animals in the dataset
            if time_accumulated == 0:
                num_animals = int(len(eeg_and_emg[0]) / 3)

            # Increment time counter
            time_accumulated += diff_time
            time_stamp_array = np.append(time_stamp_array, time_array)
            time_accumulated_array = np.append(time_accumulated_array, time_accumulated)

            # Make a list of lists for each animal (and the 3 channels for the animal)
            if not eeg_emg_lists:
                eeg_emg_lists = [[] for i in range(num_animals * 3)]

            # Append the data for each run into the list of lists
            for i in range(num_animals * 3):
                eeg_emg_lists[i].append(eeg_and_emg[:, i])

            if time_accumulated > time_run:
                break

    # Create empty concatenated list
    concatenated_list = [[] for j in range(num_animals * 3)]

    # Fill in the concatenated list
    for j in range(num_animals * 3):
        for sublist in eeg_emg_lists[j]:
            for item in sublist:
                concatenated_list[j].append(item)

    # Return needed values
    return concatenated_list, time_stamp_array, time_accumulated_array


def makefiles(path_route="/Volumes/raw_data/2023", time_per_folder=0.005):

    """
    Collects EEG/EMG data from .h5 files in a directory, splits it into multiple
    files of specified duration, and saves them as EDF files in a new directory.

    Args:
        path_route (str): The path of the directory containing the raw data.
        time_per_folder (float): The maximum duration of each output file.

    Returns:
        None

    """

    # Initialize a list of paths using the make_paths() function
    needed_paths = make_paths()

    mouse_number = 1
    # Loop through each path in needed_paths
    for path in needed_paths:

        # Create the folder path where the raw data is store
        folder = Path(path_route) / path / "raw"

        # Call the makelist_of_data() function to get the mouse_data and time_data
        mouse_data, time_stamp_data, time_accumulated_data = makelist_of_data(folder)
        time_stamp_data = (time_stamp_data - time_stamp_data[0]) / (3600)

        # The number of files that we need to create is determined by the following:
        num_files_needed = int(np.ceil(time_accumulated_data[-1] / time_per_folder))

        # Initialize indices needed; find indices needed to split data
        indices_needed = []
        for i in range(1, num_files_needed):
            indices_needed.append(
                np.argmax(np.where(np.array(time_stamp_data) < i * time_per_folder))
            )

        # Set number of signals
        N_SIGNALS = 3

        # Define the channel IDs and names for the signal headers
        for mouse_number in range(int(len(mouse_data) // 3)):
            channels = [
                ch_id + N_SIGNALS * (mouse_number) for ch_id in range(N_SIGNALS)
            ]
            channel_names = [f"{channel}" for channel in channels]

            # Extract the signal data for each channel and save as an EDF file
            signal_data = np.array(mouse_data)[channels]
            signal_data = [(np.split(data, indices_needed)) for data in signal_data]

            # Make the directory and the filename
            directory = Path(EDF_ROOT) / Path(path)
            os.makedirs(directory, exist_ok=True)

            # Split up the data into different time chunks
            for time_interval in range(len(signal_data[0])):
                time_part = [signal_data[i][time_interval] for i in range(3)]

                # Create the filename and the frequency
                filename = (
                    directory / f"mouse{mouse_number + 1}_Part{time_interval + 1}"
                ).with_suffix(".edf")
                time_stamp_data_s = time_stamp_data * 3600
                fs = 1 / np.median(time_stamp_data_s[1:] - time_stamp_data_s[:-1])
                fs = int(np.round(fs))

                # Find signal headers write the .edf file
                signal_headers = highlevel.make_signal_headers(
                    channel_names, sample_frequency=fs
                )
                header = highlevel.make_header(
                    patientname=f"Animal {mouse_number}_Part {time_interval + 1}"
                )
                highlevel.write_edf(
                    filename.as_posix(),
                    np.array(time_part) / 50,
                    signal_headers,
                    header,
                )

    return None

 filename = (Path(output_dir) / f"time_period{i+1}").with_suffix(".edf")
        period_data = period_data * 3600
        fs = 1 / np.median(period_data[1:] - period_data[:-1])
        fs = int(np.round(fs))

        # Find signal headers/ Write the .edf file
       signal_headers = [{"label": col} for col in period_data.columns]


"""
# Create the path to the .h5 files
folder_path = "h5 files"
data_list = []
EEG1_Data = []
EEG2_Data = []
EMG_Data = []

number_files_needed = 5
counter = 0

# Get all of the .h5 files in the folder
for filename in os.listdir(folder_path):
    while counter < number_files_needed:
        if filename.endswith(".h5"):
            # Make the directory a directory
            file = os.path.join(folder_path, filename)
            with h5py.File(file, "r") as f:
                dataset_names = list(f.keys())
                # Get the file contents
                data = f['data'][:]

                # Separate the dataset contents into EEG and EMG data (First two: EEG and last one: EMG)
                for i in range(21):
                    if i % 3 == 0:
                        EEG1_Data.append(data[:, i])

                    elif i % 3 == 1:
                        EEG2_Data.append(data[:, i])

                    else:
                        EMG_Data.append(data[:, i])

        # Make sure to iterate only until number_files_needed is reached
        counter += 1


# Plot through all of the EEG1 Data
counter_EEG1= 1
for eeg1_data in EEG1_Data:
    plt.plot(eeg1_data)
    plt.title(f"EEG Data #1 Counter: Trial #{counter_EEG1}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
    counter_EEG1 +=1


# Plot through all of the EEG2 Data
counter_EEG2= 1
for eeg2_data in EEG2_Data:
    plt.plot(eeg2_data)
    plt.title(f"EEG Data #2 Counter: Trial #{counter_EEG2}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
    counter_EEG2 +=1

# Plot through all of the EMG Data
counter_EMG = 1
for emg_data in EMG_Data:
    plt.plot(emg_data)
    plt.title(f"EMG Data Counter: Trial #{counter_EMG}")
    plt.xlabel("Time")
    plt.ylabel ("Amplitude")
    plt.show()
    counter_EMG += 1


# Create the signal Headers for the .edf file
signal_headers = []
for count in range(number_files_needed):
    signal_headers.append({"label": f"Signal {count}", "dimension": "mV", "sample_rate": 100})



# Write the .edf file for the EEG Trial #1
edf_file_EEG1 = pyedflib.EdfWriter("EEG1_concatenated.edf", len(EEG1_Data))
edf_file_EEG1.setSignalHeaders(signal_headers)
edf_file_EEG1.writeSamples(EEG1_Data)
edf_file_EEG1.close()

# Write the .edf file for the EEG Trial #2
edf_file_EEG2 = pyedflib.EdfWriter("EEG2_concatenated.edf", len(EEG2_Data))
edf_file_EEG2.setSignalHeaders(signal_headers)
edf_file_EEG2.writeSamples(EEG2_Data)
edf_file_EEG2.close()

# Write the .edf file for the EMG Trial
edf_file_EMG = pyedflib.EdfWriter("EMG_concatenated.edf", len(EMG_Data))
edf_file_EMG.setSignalHeaders(signal_headers)
edf_file_EMG.writeSamples(EMG_Data)
edf_file_EMG.close()

"""


if __name__ == "__main__":
    makefiles()
