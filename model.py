from extract import DataType, DEFAULT_DATA_ROOT_DIR, parse_data_files, parse_data_file, data_map
from plot import ImuAxis, TIME_STEP_SEC

from typing import List
import numpy as np

def compute_sleep_probabilities(person_ids: List[str] = None, test_data=None):
    """
    -1: Unspecified
    0: Wake
    1: N1
    2: N2
    3: N3 (Deep)
    5: REM 

    The probabilities list:
    * Row: time index
    * Column: "bin" or the integer value corresponding to the sleep state
    * (Row, Column) --> the probability of that sleep state at that time index

    We are only interested in 0, 1, 2, 3, and 5

    # # Cumulative distribution function, sum over columns
    # cdf = np.cumsum(np.array(probabilities), axis=1)

    # # Normalize each row (optional)
    # row_sums = cdf[:, -1]  # Last column (sum of each row)
    # cdf = cdf / row_sums[:, np.newaxis]  # Normalize each row to sum to 1
    # return cdf
    """
    if test_data is None:
        parse_data_files(DEFAULT_DATA_ROOT_DIR, person_ids=person_ids, key=DataType.STATE)

        data = []
        for datum in zip(*(c.data for c in data_map[DataType.STATE.name])):
            data.append(datum)
    else:
        data = test_data

    data = np.array(data)
    probabilities = []

    for time_index in range(data.shape[1]):
        col = data[:, time_index]

        # Filter out the unspecified state
        col = np.array(list(filter(lambda x: x != -1, col)))

        counts = np.bincount(col.astype(int), minlength=6)  # 0 to 5

        # Merge the first and second results, since this represents "core" sleep
        # which merges N1 and N2)
        core_count = counts[1] + counts[2]
        revised_counts = [counts[0], core_count]
        for i in range(3, len(counts)):
            revised_counts.append(counts[i])

        revised_counts = np.array(revised_counts)
        # Normalize to get probabilities
        probs = revised_counts / revised_counts.sum()
        # probs = counts / counts.sum()

        probabilities.append(probs)

    return np.array(probabilities)

def find_sleep_probabilities(prob_mat, time, start_time):
    # Get the probabilities from the row corresponding to the nearest time
    time_index = int(time / TIME_STEP_SEC) - int(start_time / TIME_STEP_SEC)

    if time_index < len(prob_mat):
        return prob_mat[time_index]
    return None

def find_best_time(prob_mat, target_probs: List[float], sleep_states: List[int]):
    """
    Find the first time index where the probabilities of sleep_states yield the
    minimum cumulative difference with target_probs
    """
    total_diffs = []

    # Loop over each time step and calculate the total difference for all values
    # row in the cdf represents a time step
    for row in range(prob_mat.shape[0]):
        total_diff = 0
        for j, sleep_state in enumerate(sleep_states):
            diff = np.abs(prob_mat[row, sleep_state] - target_probs[j])
            total_diff += diff
        total_diffs.append(total_diff)

    # Corresponds to the index with the minimum total difference
    return np.argmin(total_diffs) * TIME_STEP_SEC

def pretty_print(arr, decimals=4):
    """ Pretty prints a 2D NumPy array with aligned columns. """
    formatted_rows = [
        " | ".join(f"{val:.{decimals}f}" for val in row)
        for row in arr
    ]
    print("\n".join(formatted_rows))

def find_state_mat_probabilities(person_id=None):
    """
    Finds the values to use for the sleep stage probabilities
    in the state transition matrix
    """
    if person_id is None:
        parse_data_files(DEFAULT_DATA_ROOT_DIR, key=DataType.STATE)
    else:
        parse_data_file(DEFAULT_DATA_ROOT_DIR, person_id, type_key=DataType.STATE)
    
    data = [c.data for c in data_map[DataType.STATE.name]]

    # data = [[0, 1, 2, 3, 5],
    #         [3, 2, 1, 5, 0],
    #         [5, 3, 0, 2, 1]]

    # Keeps track of frequency of transitions
    freq_mat = np.zeros((6, 6))

    for patient_data in data:
        filtered_data = list(filter(lambda x: x != -1 and x != 4, patient_data))
        current_state = filtered_data[0]
        for i in range(1, len(filtered_data)):
            next_state = filtered_data[i]
            freq_mat[current_state, next_state] += 1
            current_state = next_state

    # State transitions are unique, so there will be no multiple counting
    freq_mat[1, :] += freq_mat[2, :]
    freq_mat = np.delete(freq_mat, 2, axis=0)  # Remove row 2

    # Merge columns: Sum column 1 and column 2 into column 1
    freq_mat[:, 1] += freq_mat[:, 2]
    freq_mat = np.delete(freq_mat, 2, axis=1)  # Remove column 2

    # Normalize along rows
    row_sums = np.sum(freq_mat, axis=1, keepdims=True)
    prob_mat = np.divide(freq_mat, row_sums, where=row_sums != 0)

    # pretty_print(freq_mat)
    #print(np.sum(prob_mat, axis=1))
    pretty_print(prob_mat)

def find_obs_mat_weights():
    """
    This function wants to answer:
    1. What is the average heart rate when the patient is in a 
        given sleep stage?
    2. What are the average x, y, and z axis accelerations when the 
        patient is in a given sleep stage?

    These values will be used in the weights of the observation model.
    """

    # Keys represent the states
    hr_map = {key: [0, 0] for key in [0, 1, 2, 3, 5]}

    # Each key is the state
    # Each value is a 4-tuple, with:
    # [0] -> count
    # [1], [2], [3] -> x, y, z
    imu_map = {key: [0, 0.0, 0.0, 0.0] for key in [0, 1, 2, 3, 5]}
    
    # Load all of the data from all patients
    parse_data_files(DEFAULT_DATA_ROOT_DIR, key=DataType.HEART_RATE)
    parse_data_files(DEFAULT_DATA_ROOT_DIR, key=DataType.MOTION)
    parse_data_files(DEFAULT_DATA_ROOT_DIR, key=DataType.STATE)

    state_data = []
    for c in data_map[DataType.STATE.name]:
        filtered_data = list(filter(lambda x: x != -1 and x != 4, c.data))
        state_data.append(filtered_data)

    # Loop through heart rate data, and perform time match
    for person_index in range(0, len(state_data)):
        # Process heart rate data
        hr_cont = data_map[DataType.HEART_RATE.name][person_index]
        motion_cont = data_map[DataType.MOTION.name][person_index]
        
        # Find the first positive index (the same for both heart rate and motion data)
        first_index = 0
        for index, hr_tp in enumerate(hr_cont.time):
            if hr_tp / 60 > 0:
                first_index = index
                break

        for index in range(first_index, len(hr_cont.data), TIME_STEP_SEC):
            if (index > len(state_data[person_index])):
                break

            state = state_data[person_index][index]
            hr_map[state][0] += 1
            hr_map[state][1] += hr_cont.data[index]
            
            imu_map[state][0] += 1
            imu_dp = motion_cont.data[index]
            imu_map[state][1] += imu_dp[0]
            imu_map[state][2] += imu_dp[1]
            imu_map[state][3] += imu_dp[2]

    # Perform an average of the weights

    # Each element is the average measurement, each index: wake, deep, rem, core
    hr_weights = []
    imu_x_weights = []
    imu_y_weights = []
    imu_z_weights = []

    for state in [0, 3, 5]:
        d_hr = hr_map[state]
        hr_weights.append(float(d_hr[1]) / d_hr[0])

        d_imu = imu_map[state]
        imu_x_weights.append(d_imu[1] / float(d_imu[0]))
        imu_y_weights.append(d_imu[2] / float(d_imu[0]))
        imu_z_weights.append(d_imu[3] / float(d_imu[0]))

    # Merge 1 and 2 into "core" sleep category
    core_hr_count = hr_map[1][0] + hr_map[2][0]
    core_hr = hr_map[1][1] + hr_map[2][1]
    hr_weights.append(float(core_hr) / core_hr_count)

    core_imu_count = imu_map[1][0] + imu_map[2][0]
    core_imu_x = imu_map[1][1] + imu_map[2][1]
    core_imu_y = imu_map[1][2] + imu_map[2][3]
    core_imu_z = imu_map[1][3] + imu_map[2][3]

    imu_x_weights.append(core_imu_x / core_imu_count)
    imu_y_weights.append(core_imu_y / core_imu_count)
    imu_z_weights.append(core_imu_z / core_imu_count)

    print(f"Heart Rate Weights: {hr_weights}")
    print(f"IMU x-axis: {imu_x_weights}")
    print(f"IMU y-axis: {imu_y_weights}")
    print(f"IMU z-axis: {imu_z_weights}")

def main():
    # test_data = [
    #     [1, 2, -1, 3, 5],  # List 1
    #     [0, 2, 3, 4, 0],  # List 2
    #     [1, 1, 2, 2, 3],  # List 3
    # ]

    # #prob_mat = compute_sleep_probabilities(test_data=test_data)
    # prob_mat = compute_sleep_probabilities()

    # # Target probs have to add to 1
    # # Index 4 must always be 0 - still need to pad
    # # represents: wake, core, deep, <invalid>, rem
    # target_probs = [0.2, 0.2, 0.2, 0, 0.2]
    # sleep_states = [i for i in range(5)]

    # index = find_best_time(prob_mat, target_probs, sleep_states)
    # print(index)

    # find_state_mat_probabilities("4314139")
    # find_state_mat_probabilities()
    find_obs_mat_weights()

if __name__ == "__main__":
    main()
