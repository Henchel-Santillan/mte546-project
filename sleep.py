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

R_mat = []

############################## SENSOR MODELS and NOISE #################################
# Assume zero mean Gaussian for noise
def __gaussian_noise(mean=0, std_dev=0, num_samples=1):
    return np.random.normal(mean, std_dev, num_samples)

def hr_sensor_model_ekf(time, has_noise=True):
    est_hr = 2.855e-08 * time ** 2 - 0.0014 * time + 73.0994
    if has_noise:
        est_hr += __gaussian_noise(0, np.sqrt(R_mat[0, 0]))
    return est_hr

def imu_sensor_model_ekf(time, axis: ImuAxis, has_noise=True):
    match axis:
        case ImuAxis.X_AXIS:
            est_imu = -7.124e-17 * (time ** 4) + 4.18e-12 * (time ** 3) - 7.583e-08 * (time ** 2) + 0.0004 * time + 1.683e-07
            std_dev = np.sqrt(R_mat[1, 1])
        case ImuAxis.Y_AXIS:
            est_imu = -5.469e-17 * (time ** 4) + 3.283e-12 * (time ** 3) - 5.88e-08 * (time ** 2) + 0.0003 * time + 1.205e-07
            std_dev = np.sqrt(R_mat[2, 2])
        case ImuAxis.Z_AXIS:
            est_imu = -1.011e-16 * (time ** 4) + 5.957e-12 * (time ** 3) - 1.084e-07 * (time ** 2) - 0.0006 * time + 2.473e-07
            std_dev = np.sqrt(R_mat[3, 3])
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
hr_weights = [1, 0.7, 0.5, 1.3, 0, 0 ,0]
base_imu_weights = [1.6, 0.8, 0.1, 1.3, 0, 0, 0]

def HJacobian(x, time, *args):
    """
    Function that computes the HJacobian.
    Take partial derivative of measurements with respect to the states.
    Yields m x n matrix, m = number of measurements, n = number of states
    """
    hr = hr_weights * hr_sensor_model_ekf(time)
    imu_x = base_imu_weights * imu_sensor_model_ekf(time, ImuAxis.X_AXIS)
    imu_y = base_imu_weights * imu_sensor_model_ekf(time, ImuAxis.Y_AXIS)
    imu_z = base_imu_weights* imu_sensor_model_ekf(time, ImuAxis.Z_AXIS)
    return np.array([hr, imu_x, imu_y, imu_z])

def Hx(x, time, *args):
    """
    Given state vector x, return the corresponding measurement for this step
    """
    # stage_nums = [i for i in range(5)]

    # # Extract the sleep stage probabilities from the state vector
    # # Pad 3rd index with "0", since "4" doesn't correspond to any sleep state
    # sleep_probabilities = [x[0], x[1], x[3], 0, x[2]]
    # time = find_best_time(sleep_state_prob_mat, sleep_probabilities, stage_nums)

    # Columns: Wake, Core, Deep, REM, use the sleep probabilities as weights
    # Time-varying linear weighted sum approach, may need tuning or a different approach
    x = x.reshape(1, -1)
    sleep_probabilities = np.array([x[0, 0], x[0, 1], x[0, 3], x[0, 2], 1, 1, 1])

    z1 = np.dot(hr_weights * hr_sensor_model_ekf(time), sleep_probabilities)
    z2 = np.dot(base_imu_weights * imu_sensor_model_ekf(time, ImuAxis.X_AXIS), sleep_probabilities)
    z3 = np.dot(base_imu_weights * imu_sensor_model_ekf(time, ImuAxis.Y_AXIS), sleep_probabilities)
    z4 = np.dot(base_imu_weights * imu_sensor_model_ekf(time, ImuAxis.Z_AXIS), sleep_probabilities)

    # Return the vector of expected measurements
    return np.array([z1, z2, z3, z4]).reshape(4, 1)

def get_data_at_time(time: int):
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

def normalize_sleep_probabilities(x):
    sleep_states = x[:4]
    sleep_states /= np.sum(sleep_states)
    print(sleep_states)
    x[:4] = sleep_states
    return x

def main():
    # Extract the data from the .txt files
    # parse_data_files(DEFAULT_DATA_ROOT_DIR)
    #print(data_map[DataType.MOTION.name][0].data)
    parse_data_file(DEFAULT_DATA_ROOT_DIR, "4314139")
    # plot_heartrate_data(DEFAULT_PERSON_INDEX)
    # plot_motion_data(DEFAULT_PERSON_INDEX)

    # Initialize EKF
    Q_mat = np.array([
        [0.1, 0, 0, 0, 0, 0, 0],
        [0, 0.1, 0, 0, 0, 0, 0],
        [0, 0, 0.5, 0, 0, 0, 0],
        [0, 0, 0, 0.1, 0, 0, 0],
        [0, 0, 0, 0, 0.5, 0, 0],
        [0, 0, 0, 0, 0, 0.5, 0],
        [0, 0, 0, 0, 0, 0, 0.5],
    ])

    # R matrix is diagonal, assumes measurement noises to be independent of one another
    # The diagonals are populated by the variances of each measurement
    var_hr, var_imu_x, var_imu_y, var_imu_z = get_measurement_variances(DEFAULT_PERSON_INDEX)
    global R_mat
    R_mat = np.array([
        [var_hr, 0, 0, 0],
        [0, var_imu_x, 0, 0],
        [0, 0, var_imu_y, 0],
        [0, 0, 0, var_imu_z],
    ])

    ekf = EKF(NUM_STATES, NUM_MEASUREMENTS)

    alpha_rem = 1.2  # rate at which REM increases 'wake score'
    beta_core = 3  # rate at which core sleep decreases 'wake score'
    gamma = 0.5  # decay the weight of previous values on current computation (smoothing factor)

    # A = np.array([
    #     [0.85, 0.10, 0.05, 0.00, 0, 0, 0],
    #     [0.10, 0.75, 0.10, 0.05, 0, 0, 0],
    #     [0.00, 0.20, 0.75, 0.05, 0, 0, 0],
    #     [0.00, 0.10, 0.10, 0.80, 0, 0, 0],
    #     [0, 0, alpha_rem, 0, 1, 0, -beta_core],  # Wake Score Eq
    #     [0, 0, gamma, 0, 0, 1 - gamma, 0],  # REM Duration Eq
    #     [0, gamma, 0, 0, 0, 0, 1 - gamma]   # Core Duration Eq
    # ])  # 7x7

    # A = np.array([
    #     [0.6505, 0.3495, 0, 0, 0, 0, 0],
    #     [0.0518, 0.9314, 0.0067, 0.01, 0, 0, 0],
    #     [0.0312, 0.0312, 0, 0.9375, 0, 0, 0],
    #     [0.0123, 0.0123, 0.9755, 0, 0, 0, 0],
    #     [0, 0, alpha_rem, 0, 1, 0, -beta_core],  # Wake Score Eq
    #     [0, 0, gamma, 0, 0, 1 - gamma, 0],  # REM Duration Eq
    #     [0, gamma, 0, 0, 0, 0, 1 - gamma]   # Core Duration Eq
    # ])  # 7x7

    A = np.array([
        [0.6505, 0.3495, 0, 0, 0, 0, 0],
        [0.1018, 0.7514, 0.1367, 0.01, 0, 0, 0],
        [0.0312, 0.3012, 0.03, 0.6375, 0, 0, 0],
        [0.0123, 0.3123, 0.4755, 0.2, 0, 0, 0],
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

    start_time = 930
    final_time = 14400 # 4 hours in seconds

    curr_time = start_time
    while curr_time < final_time:
        patient_data = get_data_at_time(curr_time)

        curr_z = np.array(patient_data[:-1])  # exclude ground truth state
        curr_z = curr_z.reshape(-1, 1)

        ekf.predict_update(curr_z, HJacobian, Hx, args=(curr_time), hx_args=(curr_time))
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



##### EKF.py
# def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):
#         """ Performs the predict/update innovation of the extended Kalman
#         filter.

#         Parameters
#         ----------

#         z : np.array
#             measurement for this step.
#             If `None`, only predict step is perfomed.

#         HJacobian : function
#            function which computes the Jacobian of the H matrix (measurement
#            function). Takes state variable (self.x) as input, along with the
#            optional arguments in args, and returns H.

#         Hx : function
#             function which takes as input the state variable (self.x) along
#             with the optional arguments in hx_args, and returns the measurement
#             that would correspond to that state.

#         args : tuple, optional, default (,)
#             arguments to be passed into HJacobian after the required state
#             variable.

#         hx_args : tuple, optional, default (,)
#             arguments to be passed into Hx after the required state
#             variable.

#         u : np.array or scalar
#             optional control vector input to the filter.
#         """
#         #pylint: disable=too-many-locals

#         if not isinstance(args, tuple):
#             args = (args,)

#         if not isinstance(hx_args, tuple):
#             hx_args = (hx_args,)

#         if np.isscalar(z) and self.dim_z == 1:
#             z = np.asarray([z], float)
#         F = self.F
#         B = self.B
#         P = self.P
#         Q = self.Q
#         R = self.R
#         x = self.x

#         H = HJacobian(x, *args)

#         # predict step
#         x = dot(F, x) + dot(B, u)
#         P = dot(F, P).dot(F.T) + Q

#         # save prior
#         self.x_prior = np.copy(self.x)
#         self.P_prior = np.copy(self.P)

#         # update step
#         PHT = dot(P, H.T)

#         # print(f"PHT: {PHT.shape}")
#         self.S = dot(H, PHT) + R
#         self.SI = linalg.inv(self.S)
#         self.K = dot(PHT, self.SI)
#         # print(f"K: {self.K.shape}")

#         self.y = z - Hx(x, *hx_args)
#         # print(f"z_val: {z}")

#         # print(f"z: {z.shape}")
#         # print(f"Hx: {Hx(x, *hx_args).shape}")

#         # print(f"y: {self.y.shape}")
#         # print(f"Hx_val: {Hx(x, *hx_args)}")
#         # print(f"y_val: {self.y}")
#         # print(f"K_dot_y: {dot(self.K, self.y).shape}")
#         # print(f"x: {x.shape}")


#         self.x = x.reshape(-1,1) + dot(self.K, self.y)  # HEEEREREERERE

#         I_KH = self._I - dot(self.K, H)
#         self.P = dot(I_KH, P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

#         # save measurement and posterior state
#         self.z = deepcopy(z)
#         self.x_post = self.x.copy()
#         self.P_post = self.P.copy()

#         # set to None to force recompute
#         self._log_likelihood = None
#         self._likelihood = None
#         self._mahalanobis = None
