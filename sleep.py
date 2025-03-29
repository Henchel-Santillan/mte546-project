#from filterpy import ExtendedKalmanFilter as EKF
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from enum import Enum
import csv
import os

NUM_STATES = 4 # awake, core sleep = N1/N2, REM, Deep sleep = N3
NUM_MEASUREMENTS = 3  # HRV, HR, motion (duration)
TIME_STEP = 30  # time step our simulation loops through (in seconds)
DEFAULT_PERSON_INDEX = 0

## MEASUREMENTS
class DataType(Enum):
    MOTION = 1,
    HEART_RATE = 2,
    STATE = 3,  # ground truth data

class ImuAxis(Enum):
    X_AXIS = 1
    Y_AXIS = 2
    Z_AXIS = 3

class DataContainer:
    def __init__(self):
        self.time = []
        self.data = []

# Change this value to wherever the data is saved
DEFAULT_DATA_ROOT_DIR=f"{os.getenv("USERPROFILE")}/Downloads/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"

# Data map has keys "MOTION" and "HEART_RATE", and value of DataContainer
# data_map[key].time gives a 1D list of timestamps
# data_map[key].data gives a 1D list of data
# For MOTION data, each element in the list is a tuple for the accelerations measured in g (x, y, z)
# For HEART_RATE data,hear each element in the list is 
data_map = {DataType.MOTION.name: [], 
            DataType.HEART_RATE.name: [],
            DataType.STATE.name: []}

def __extract_data(file, data_type: DataType) -> DataContainer:
    container = DataContainer()
    with open(file, mode="r") as f:
        # Motion data is space separated, heart rate data is comma-separated
        delim = "," if data_type == DataType.HEART_RATE else " "
        csv_file = csv.reader(f, delimiter=delim)
        for line in csv_file:
            # Only add the data if time is greater than 0, which represents
            # when the person is no longer awake
            if float(line[0]) > 0:
                # Record the timestamp
                # Record the data
                match data_type:
                    case DataType.MOTION:
                        # x, y, z -> pack into a tuple
                        container.time.append(float(line[0]))
                        container.data.append((float(line[1]), float(line[2]), float(line[3])))
                    case DataType.HEART_RATE:
                        # Just BPM data, append directly
                        container.time.append(float(line[0]))
                        container.data.append(int(line[1]))
                    case DataType.STATE:
                        ground_truth = int(line[1])
                        if ground_truth > 0:
                            # Data uses -1 to represent unknown state
                            container.time.append(float(line[0]))
                            container.data.append(ground_truth)
                    case _:
                        continue
    return container

def parse_data_file(data_root_dir: str, person_id: str) -> None:
    type_dict = {
        "motion" : DataType.MOTION,
        "heart_rate" : DataType.HEART_RATE,
        "labels" : DataType.STATE 
    }

    for key, value in type_dict.items():
        data_path = os.path.join(data_root_dir, key)
        for root, _, files in os.walk(data_path):
            for file in files:
                if person_id in file:
                    container = __extract_data(os.path.join(root, file), value)
                    data_map[value.name].append(container)

# def parse_data_files(data_root_dir: str) -> None:
#     type_dict = {
#         "motion" : DataType.MOTION,
#         "heart_rate" : DataType.HEART_RATE,
#         "labels" : DataType.STATE 
#     }

#     for key, value in type_dict.items():
#         data_path = os.path.join(data_root_dir, key)
#         for root, _, files in os.walk(data_path):
#             for file in files:
#                 container = __extract_data(os.path.join(root, file), value)
#                 data_map[value.name].append(container)

def preprocess_data() -> None:
    pass


############################## SENSOR MODELS #####################################
# Sensor models: # use robust least squares to fit heart rate data
def sensor_model_hr(person_i: int):
    hr_time = data_map[DataType.HEART_RATE.name][person_i].time
    hr_data = data_map[DataType.HEART_RATE.name][person_i].data

    # Regressors / intercept
    X = np.column_stack([np.ones_like(hr_time), hr_time, [time ** 2 for time in hr_time]])

    # Fit robust model (HuberT norm)
    robust_model = sm.RLM(hr_data, X, M=sm.robust.norms.HuberT()).fit()
    print(robust_model.summary())

    # Gives -0.0006x^2 + 96.6742x + 69.1904 using 4314139
    return robust_model.predict(X)

def sensor_model_motion(person_i: int, axis: ImuAxis, should_print=False):
    motion_time = data_map[DataType.MOTION.name][person_i].time
    motion_val = data_map[DataType.MOTION.name][person_i].data

    # Regressors / intercepts
    #X = sm.add_constant(motion_time)
    X = np.column_stack([np.ones_like(motion_time), motion_time, [time ** 2 for time in motion_time]])

    # Fit robust model (HuberT norm)
    index = axis.value - 1

    model = sm.RLM([val[index] for val in motion_val], X, M=sm.robust.norms.HuberT()).fit()
    if should_print:
        print(model.summary())

    return model.predict(X)


# #################################### PLOTTING FUNCTIONS ##################################
def plot_heartrate_data(person_i: int) -> None:
    plt.figure(figsize=(8,6))
    
    # Heart rate
    heartrate_val = data_map[DataType.HEART_RATE.name][person_i].data
    heartrate_time = data_map[DataType.HEART_RATE.name][person_i].time

    plt.plot(heartrate_time, heartrate_val, label="Heartrate")
    
    # Robust fit over the heart rate data
    heart_rate_pred = sensor_model_hr(person_i)
    plt.plot(heartrate_time, heart_rate_pred, label="Heartrate Fit (Robust LS)")
    
    plt.title("Heart Rate vs. Time")
    plt.xlabel("Time [sec]")
    plt.ylabel("Heart Rate [bpm]")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_motion_data(person_i: int) -> None:
    plt.figure(figsize=(8,6))

    fig, axs = plt.subplots(3, 1)

    labels = [("ax", "ax (Robust LS)"), ("ay", "ay (Robust LS)"), ("az", "az (Robust LS)")]
    
    # Motion
    motion_time = data_map[DataType.MOTION.name][person_i].time
    motion_val = data_map[DataType.MOTION.name][person_i].data
    motion_models = (sensor_model_motion(person_i, ImuAxis.X_AXIS),
                     sensor_model_motion(person_i, ImuAxis.Y_AXIS),
                     sensor_model_motion(person_i, ImuAxis.Z_AXIS))

    for index in range(len(motion_models)):
        axs[index].plot(motion_time, [val[index] for val in motion_val], label=labels[index][0])
        axs[index].plot(motion_time, motion_models[index], label=labels[index][1])
        axs[index].set_title(f"Acceleration ({ImuAxis(index + 1).name}) vs. Time")
        axs[index].set_xlabel("Time [sec]")
        axs[index].set_ylabel("Acceleration [g]")
        axs[index].grid(True)
        axs[index].legend()

    fig.suptitle("Raw Acceleration Data")
    
    plt.tight_layout()
    plt.show()

def plot_ground_truth(person_i: int) -> None:
    plt.figure(figsize=(8,6))

    state_val = data_map[DataType.STATE.name][person_i].data
    state_time = data_map[DataType.STATE.name][person_i].time
    plt.plot(state_time, state_val, label="Ground Truth State")

def HJacobian(x, *args):
    return

def Hx(x, *args):
    return

def get_data_at_time(time: int) -> list:
    data = []
    time_arrays = [
        data_map[DataType.HEART_RATE.name][0].time,
        data_map[DataType.MOTION.name][0].time,
        data_map[DataType.STATE.name][0].time
    ]
    
    data_arrays = [
        data_map[DataType.HEART_RATE.name][0].data,
        data_map[DataType.MOTION.name][0].data,
        data_map[DataType.STATE.name][0].data
    ]
    
    for timestamps, values in zip(time_arrays, data_arrays):  # loop through all input data sources (e.g. heartrate, motion, state)
        time_arrays = np.array(time_arrays)
        valid_timestamps = timestamps[timestamps <= time]
        
        if len(valid_timestamps) == 0:  # JOSH: the case where there are no matching time stamps
            print("NOOOOOOOOOOOOOOOOOOOOOOOO")  # idk what to do, maybe fix by setting t=15 and resume at 30 afterwards?
            
        else:
            valid_timestamp_index = np.argmax(valid_timestamps)
            if type(data[valid_timestamp_index]) is tuple:  # motion is a tuple we want to unpack
                accel_data = np.asarray(data[valid_timestamp_index])  # convert accel data from 
                data = data + accel_data.tolist()
            else:
                data.append(data[valid_timestamp_index])
            
    return data  # heart_rate, motion (x, y, z), state

def main():
    # Extract the data from the .txt files
    # parse_data_files(DEFAULT_DATA_ROOT_DIR)
    #print(data_map[DataType.MOTION.name][0].data)
    parse_data_file(DEFAULT_DATA_ROOT_DIR, "4314139")
    # plot_heartrate_data(DEFAULT_PERSON_INDEX)
    plot_motion_data(DEFAULT_PERSON_INDEX)

    # Initialize EKF
    Q_mat = np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0.5]
    ])
    R_mat = np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0.5]
    ])
    
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
    
    # curr_time = 0 # or 1 idk
    
    # not_end = True
    # while not_end:
    #     patient_data = get_data_at_time(curr_time)
    #     curr_z = np.array([0, patient_data[0], patient_data[1]])  # JOSH: idt it splits up motions
    #     curr_z = curr_time.reshape(-1, 1)  # JOSH: does this work?
        
        
    #     curr_time += TIME_STEP
    #     curr_z = z[:, i]
    #     curr_z = z.reshape(-1, 1)
        
    #     ekf.predict_update(curr_z, HJacobian, Hx)
    #     posterior_state = ekf.x
    #     estimated_states.append(posterior_state)
    #     i+=1
    

if __name__ == "__main__":
    main() 

