from filterpy import ExtendedKalmanFilter as EKF
import numpy as np

from enum import Enum
import csv
import os

NUM_STATES = 4 # awake, core sleep = N1/N2, REM, Deep sleep = N3
NUM_MEASUREMENTS = 3  # HRV, HR, motion (duration)

## MEASUREMENTS
class DataType(Enum):
    MOTION = 1,
    HEART_RATE = 2,

class DataContainer:
    def __init__(self, time=[], data=[]):
        self.time = time
        self.data = data

# Change this value to wherever the data is saved
DEFAULT_DATA_ROOT_DIR=f"{os.getenv("USERPROFILE")}/Downloads/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"
data_map = dict.fromkeys([DataType.MOTION.name, DataType.HEART_RATE.name], DataContainer)

def __extract_data(file, data_type: DataType) -> None:
    container = DataContainer()
    with open(file, mode="r") as f:
        # Motion data is space separated, heart rate data is comma-separated
        delim = " " if data_type == DataType.MOTION else ","
        csv_file = csv.reader(f, delimiter=delim)
        for line in csv_file:
            container.time.append(line[0])
            match data_type:
                case DataType.MOTION:
                    # x, y, z -> pack into a tuple
                    container.data.append((line[1], line[2], line[3]))
                case DataType.HEART_RATE:
                    # Just BPM data, append directly
                    container.data.append(line[1])
                case _:
                    continue
    
    # Add the container to the data map
    data_map[data_type.name] = container

def parse_data_files(data_root_dir: str) -> None:
    types = [e for e in DataType]
    for type in types:
        data_path = os.path.join(data_root_dir, type.name.lower())
        for root, _, files in os.walk(data_path):
            for file in files:
                __extract_data(os.path.join(root, file), type)

def HJacobian(x, *args):
    return

def Hx(x, *args):
    return

def main():
    # Extract the data from the .txt files
    parse_data_files(DEFAULT_DATA_ROOT_DIR)
    print(data_map[DataType.MOTION.name].data)

    # Initialize EKF
    # Q_mat = np.array([
    #     [0.5, 0, 0],
    #     [0, 0.5, 0],
    #     [0, 0, 0.5]
    # ])
    # R_mat = np.array([
    #     [0.5, 0, 0],
    #     [0, 0.5, 0],
    #     [0, 0, 0.5]
    # ])
    
    # ekf = EKF(NUM_STATES, NUM_MEASUREMENTS)
    # ekf.x = np.array([1, 0, 0, 0]) # initial state (start awake)
    
    # ekf.P = np.array([
    #     [0.85, 0.15, 0, 0],
    #     [0.1, 0.6, 0.25, 0.05],
    #     [0.05, 0.5, 0.4, 0.05],
    #     [0, 0.7, 0.1, 0.2]
    #     ]) # model: each row represents the probability of transitioning form one state to another
    # ekf.H = None
    
    # # Main Loop
    # estimated_states = []
    # i = 0
    # z = ... # measurement readings, shape: [NUM_MEASUREMENTS, N]
    
    # while True:
    #     curr_z = z[:, i]
    #     curr_z = z.reshape(-1, 1)
        
    #     ekf.predict_update(curr_z, HJacobian, Hx)
    #     posterior_state = ekf.x
    #     estimated_states.append(posterior_state)
    #     i+=1
    

if __name__ == "__main__":
    main() 

