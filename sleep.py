from filterpy.kalman import ExtendedKalmanFilter as EKF
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from enum import Enum
from typing import List
import csv
import os

NUM_STATES = 4 # awake, core sleep = N1/N2, REM, Deep sleep = N3
NUM_MEASUREMENTS = 3  # HRV, HR, motion (duration)
TIME_STEP_SEC = 30  # time step our simulation loops through (in seconds)
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
DEFAULT_DATA_ROOT_DIR=f"{os.getenv('USERPROFILE')}/Downloads/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"

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

# The key is to parse
def parse_data_file(data_root_dir: str, person_id: str, type_key=None) -> None:
    type_dict = {
        DataType.MOTION : "motion",
        DataType.HEART_RATE : "heart_rate",
        DataType.STATE : "labels"
    }

    def append_container(data_root_dir, key, value):
        match key:
            case DataType.MOTION:
                file_name = f"{person_id}_acceleration.txt"
            case DataType.HEART_RATE:
                file_name = f"{person_id}_heartrate.txt"
            case DataType.STATE:
                file_name = f"{person_id}_labeled_sleep.txt"
        data_path = os.path.join(data_root_dir, value, file_name)
        container = __extract_data(data_path, key)
        data_map[key.name].append(container)

    if type_key is None:
        # Can just read the files directly instead of walking through the entire directory
        for key, value in type_dict.items():
            append_container(data_root_dir, key, value)
    else:
        value = type_dict[type_key]
        append_container(data_root_dir, type_key, value)

def parse_data_files(data_root_dir: str, person_ids: List[str], key=None):
    for person_id in person_ids:
        parse_data_file(data_root_dir, person_id, key)

# Returns variances in this order: (heart rate, IMU x, IMU y, IMU z)
def get_measurement_variances(person_i: int):
    var_hr = np.var(np.array(data_map[DataType.HEART_RATE.name][person_i].data))
    imu_data = data_map[DataType.MOTION.name][person_i].data

    var_imu_x = np.var(np.array([val[0] for val in imu_data]))
    var_imu_y = np.var(np.array([val[1] for val in imu_data]))
    var_imu_z = np.var(np.array([val[2] for val in imu_data]))
    return (var_hr, var_imu_x, var_imu_y, var_imu_z)

############################## SENSOR MODELS and NOISE #################################
# Assume zero mean Gaussian for noise
def __gaussian_noise(mean=0, std_dev=0, num_samples=1):
    return np.random.normal(mean, std_dev, num_samples)

def hr_sensor_model_ekf(time, r_mat, has_noise=True):
    est_hr = 2.855e-08 * time ** 2 - 0.0014 * time + 73.0994
    if has_noise:
        est_hr += __gaussian_noise(0, np.sqrt(r_mat[0, 0]))
    return est_hr

def imu_sensor_model_ekf(time, r_mat, axis: ImuAxis, has_noise=True):
    match axis:
        case ImuAxis.X_AXIS:
            est_imu = -7.124e-17 * (time ** 4) + 4.18e-12 * (time ** 3) - 7.583e-08 * (time ** 2) + 0.0004 * time + 1.683e-07
            std_dev = np.sqrt(r_mat[1, 1])
        case ImuAxis.Y_AXIS:
            est_imu = -5.469e-17 * (time ** 4) + 3.283e-12 * (time ** 3) - 5.88e-08 * (time ** 2) + 0.0003 * time + 1.205e-07
            std_dev = np.sqrt(r_mat[2, 2])
        case ImuAxis.Z_AXIS:
            est_imu = -1.011e-16 * (time ** 4) + 5.957e-12 * (time ** 3) - 1.084e-07 * (time ** 2) - 0.0006 * time + 2.473e-07
            std_dev = np.sqrt(r_mat[3, 3])
        case _:
            return
    
    if has_noise:
        est_imu += __gaussian_noise(0, std_dev)
    return est_imu

# #################################### PLOTTING FUNCTIONS ##################################
def plot_heartrate_data(person_i: int) -> None:
    """
    MODEL:
    2.855e-08x^2 - 0.0014x + 73.0994
    """
    def gen_sensor_model_hr(person_i: int):
        hr_time = data_map[DataType.HEART_RATE.name][person_i].time
        hr_data = data_map[DataType.HEART_RATE.name][person_i].data

        # Regressors / intercept
        X = np.column_stack([np.ones_like(hr_time), hr_time, [time ** 2 for time in hr_time]])

        # # Fit robust model (HuberT norm)
        # robust_model = sm.RLM(hr_data, X, M=sm.robust.norms.HuberT()).fit()

        # Use bisquare robust LS
        robust_model = sm.RLM(hr_data, X, M=sm.robust.norms.TukeyBiweight()).fit()
        print(robust_model.summary())

        # Gives -0.0006x^2 + 96.6742x + 69.1904 using 4314139
        return robust_model.predict(X)

    plt.figure(figsize=(8,6))
    
    # Heart rate
    heartrate_val = data_map[DataType.HEART_RATE.name][person_i].data
    heartrate_time = data_map[DataType.HEART_RATE.name][person_i].time

    plt.plot(heartrate_time, heartrate_val, label="Heartrate")
    
    # Robust fit over the heart rate data
    heart_rate_pred = gen_sensor_model_hr(person_i)
    plt.plot(heartrate_time, heart_rate_pred, label="Heartrate Fit (Robust LS)")
    
    plt.title("Heart Rate vs. Time")
    plt.xlabel("Time [sec]")
    plt.ylabel("Heart Rate [bpm]")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_motion_data(person_i: int) -> None:
    """
    MODELS:
    ax: -7.124e-17x^4 + 4.18e-12x^3 - 7.583e-08x^2 + 0.0004x + 1.683e-07
    ay: -5.469e-17x^4 + 3.283e-12x^3 - 5.88e-08x^2 + 0.0003x + 1.205e-07
    az: 1.011e-16x^4 - 5.957e-12x^3 + 1.084e-07  - 0.0006x - 2.473e-07
    """
    def gen_sensor_model_motion(person_i: int, axis: ImuAxis, should_print=False):
        motion_time = data_map[DataType.MOTION.name][person_i].time
        motion_val = data_map[DataType.MOTION.name][person_i].data

        # Regressors / intercepts
        #X = sm.add_constant(motion_time)
        X = np.column_stack([np.ones_like(motion_time), 
                            motion_time, 
                            [time ** 2 for time in motion_time],
                            [time ** 3 for time in motion_time],
                            [time ** 4 for time in motion_time]])

        # Fit robust model (HuberT norm)
        index = axis.value - 1

        #model = sm.RLM([val[index] for val in motion_val], X, M=sm.robust.norms.HuberT()).fit()
        model = sm.RLM([val[index] for val in motion_val], X, M=sm.robust.norms.TukeyBiweight()).fit()
        if should_print:
            print(model.summary())

        return model.predict(X)

    plt.figure(figsize=(8,6))

    fig, axs = plt.subplots(3, 1)

    labels = [("ax", "ax (Robust LS)"), ("ay", "ay (Robust LS)"), ("az", "az (Robust LS)")]
    
    # Motion
    motion_time = data_map[DataType.MOTION.name][person_i].time
    motion_val = data_map[DataType.MOTION.name][person_i].data
    motion_models = (gen_sensor_model_motion(person_i, ImuAxis.X_AXIS, should_print=True),
                     gen_sensor_model_motion(person_i, ImuAxis.Y_AXIS, should_print=True),
                     gen_sensor_model_motion(person_i, ImuAxis.Z_AXIS, should_print=True))

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
    
    
def plot_ekf_states(states: np.array, ground_truth_states: list, start_time: float, final_time: float):
    labels = [
        "P(awake)",
        "P(core)",
        "P(rem)",
        "P(deep)",
        "Wake Score",
        "T_rem",
        "T_core"
    ]
    time_range = np.arange(start_time, final_time + TIME_STEP_SEC, TIME_STEP_SEC)
    
    fig, axes = plt.subplots(3, 1, figsize=(16,8))
    
    # EKF outputs
    for state_i in labels:
        axes[0].plot(time_range[: len(time_range)-1], states[state_i, :], label=labels[state_i])
    
    axes[0].set_title("Predicted States")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("State")
    axes[0].legend()
    axes[0].grid(True)
    
    
    # Ground truth state vs predicted state (high %)
    
    
    
    # Sleep score 
    
    plt.show()


########################### EKF FUNCTIONS ##################################
def HJacobian(x, *args):
    """
    Function that computes the HJacobian
    """
    return

def Hx(x, *args):
    """
    Given state vector x, return what the measurement 
    corresponding to t
    """
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
    plot_heartrate_data(DEFAULT_PERSON_INDEX)
    plot_motion_data(DEFAULT_PERSON_INDEX)

    # Initialize EKF
    Q_mat = np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0.5]
    ])

    # R matrix is diagonal, assumes measurement noises to be independent of one another
    # The diagonals are populated by the variances of each measurement
    var_hr, var_imu_x, var_imu_y, var_imu_z = get_measurement_variances(DEFAULT_PERSON_INDEX)
    R_mat = np.array([
        [var_hr, 0, 0, 0],
        [0, var_imu_x, 0, 0],
        [0, 0, var_imu_y, 0],
        [0, 0, 0, var_imu_z],
    ])
    
    ekf = EKF(NUM_STATES, NUM_MEASUREMENTS)
    
    alpha_rem = 5  # rate at which REM increases 'wake score'
    beta_core = 3  # rate at which core sleep decreases 'wake score'
    gamma = 0.1  # decay the weight of previous values on current computation (smoothing factor)
    
    A = np.array([
        [0.85, 0.10, 0.05, 0.00, 0, 0, 0],
        [0.10, 0.75, 0.10, 0.05, 0, 0, 0],
        [0.00, 0.20, 0.75, 0.05, 0, 0, 0],
        [0.00, 0.10, 0.10, 0.80, 0, 0, 0],
        [0, 0, alpha_rem, 0, 1, 0, -beta_core],  # Wake Score Eq
        [0, 0, gamma, 0, 0, 1 - gamma, 0],  # REM Duration Eq
        [0, gamma, 0, 0, 0, 0, 1 - gamma]   # Core Duration Eq
    ])  # 7x7
    
    ekf.F = A
    ekf.P = Q_mat
    ekf.R = R_mat
    ekf.Q = Q_mat
    ekf.H = None
    
    ekf.x = np.array([
        1.0,  # P(awake)
        0,  # P(core)
        0,  # P(rem)
        0,  # P(deep)
        0,  # wake_score
        0,  # time in rem (keep track of time inside because our states are probabilisitic, meaning we actually don't know if the person has exited a state to reset it manually ourselves)
        0   # time in core
    ])  # 1x7
    
    # # ekf.x = np.array([0, 1, 0, 0, 0]) # initial state (start awake): [wake_score, initial_state, x_accel, y_accel, z_accel]
    # ekf.x = np.array([0, 1.0, 0, 0, 0]) # initial state (start awake): [wake_score, awake_state, core_sleep, rem_sleep, deep_sleep]
    
    # ekf.P = np.array([
    #     [0.85, 0.15, 0, 0],
    #     [0.1, 0.6, 0.25, 0.05],
    #     [0.05, 0.5, 0.4, 0.05],
    #     [0, 0.7, 0.1, 0.2]
    #     ]) # model: each row represents the probability of transitioning form one state to another
    # ekf.H = None
    
    # # Main Loop
    
    estimated_states = []  # store them in an array so we can see how it evolves
    ground_truth_states = []
    # i = 0
    # z = ... # measurement readings, shape: [NUM_MEASUREMENTS, N]
    
    start_time = 15 # or 1 idk
    final_time = 14400 # 4 hours in seconds
    
    curr_time = start_time
    while curr_time < final_time:
        patient_data = get_data_at_time(curr_time)
        
        curr_z = np.array(patient_data[:-1])  # exclude ground truth state
        curr_z = curr_time.reshape(-1, 1)
        
        ekf.predict_update(curr_z, HJacobian, Hx)
        posterior_state = ekf.x
        
        # record states for plotting
        estimated_states.append(posterior_state)
        ground_truth_states.append(patient_data[-1])
        
        
        curr_time += TIME_STEP_SEC
    #     i+=1
    
    # plot_ekf_states(
    #     np.hstack(estimated_states),
    #     ground_truth_states,
    #     start_time,
    #     final_time
    # )
    

if __name__ == "__main__":
    main() 
