from filterpy.kalman import ExtendedKalmanFilter as EKF
from extract import *
from plot import *

import numpy as np

# from labels:
# 0: awake
# 1: n1
# 2: n2
# 3: n3 (deep)
# 5: rem
NUM_STATES = 7 # awake, core sleep = N1/N2, REM, Deep sleep = N3
NUM_MEASUREMENTS = 3  # HRV, HR, motion (duration)
DEFAULT_PERSON_INDEX = 0

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

# Returns variances in this order: (heart rate, IMU x, IMU y, IMU z)
def get_measurement_variances(person_i: int):
    var_hr = np.var(np.array(data_map[DataType.HEART_RATE.name][person_i].data))
    imu_data = data_map[DataType.MOTION.name][person_i].data

    var_imu_x = np.var(np.array([val[0] for val in imu_data]))
    var_imu_y = np.var(np.array([val[1] for val in imu_data]))
    var_imu_z = np.var(np.array([val[2] for val in imu_data]))
    return (var_hr, var_imu_x, var_imu_y, var_imu_z)

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
        data_map[DataType.HEART_RATE.name][DEFAULT_PERSON_INDEX].time,
        data_map[DataType.MOTION.name][DEFAULT_PERSON_INDEX].time,
        data_map[DataType.STATE.name][DEFAULT_PERSON_INDEX].time
    ]
    
    data_arrays = [
        data_map[DataType.HEART_RATE.name][DEFAULT_PERSON_INDEX].data,
        data_map[DataType.MOTION.name][DEFAULT_PERSON_INDEX].data,
        data_map[DataType.STATE.name][DEFAULT_PERSON_INDEX].data
    ]
    
    for timestamps, values in zip(time_arrays, data_arrays):  # loop through all input data sources (e.g. heartrate, motion, state)
        timestamps = np.array(timestamps)
        valid_timestamps = timestamps[timestamps <= time]

        if len(valid_timestamps) == 0:  # JOSH: the case where there are no matching time stamps
            print("NOOOOOOOOOOOOOOOOOOOOOOOO")  # idk what to do, maybe fix by setting t=15 and resume at 30 afterwards? this should never run
            print(time)
            print(timestamps)

        else:
            valid_timestamp_index = np.argmax(valid_timestamps)
            if type(values[valid_timestamp_index]) is tuple:  # motion is a tuple we want to unpack
                accel_data = np.asarray(values[valid_timestamp_index])  # convert accel data from 
                data = data + accel_data.tolist()
            else:
                data.append(values[valid_timestamp_index])

    return data  # heart_rate, motion (x, y, z), state

def main():
    # Extract the data from the .txt files
    # parse_data_files(DEFAULT_DATA_ROOT_DIR)
    #print(data_map[DataType.MOTION.name][0].data)
    # parse_data_file(DEFAULT_DATA_ROOT_DIR, "4314139")
    # plot_heartrate_data(DEFAULT_PERSON_INDEX)
    # plot_motion_data(DEFAULT_PERSON_INDEX)

    # Initialize EKF
    Q_mat = np.array([
        [0.5, 0, 0, 0, 0, 0, 0],
        [0, 0.5, 0, 0, 0, 0, 0],
        [0, 0, 0.5, 0, 0, 0, 0],
        [0, 0, 0, 0.5, 0, 0, 0],
        [0, 0, 0, 0, 0.5, 0, 0],
        [0, 0, 0, 0, 0, 0.5, 0],
        [0, 0, 0, 0, 0, 0, 0.5],
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


    start_time = 930 # or 1 idk
    final_time = 14400 # 4 hours in seconds

    curr_time = start_time
    while curr_time < final_time:
        patient_data = get_data_at_time(curr_time)

        curr_z = np.array(patient_data[:-1])  # exclude ground truth state
        curr_z = curr_z.reshape(-1, 1)

        ekf.predict_update(curr_z, HJacobian, Hx)
        posterior_state = ekf.x

        # record states for plotting
        estimated_states.append(posterior_state)
        ground_truth_states.append(patient_data[-1])


        curr_time += TIME_STEP_SEC

    plot_ekf_states(
        np.hstack(estimated_states),
        ground_truth_states,
        start_time,
        final_time
    )

if __name__ == "__main__":
    main()
