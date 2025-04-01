from extract import DataType, DEFAULT_DATA_ROOT_DIR, parse_data_files, parse_data_file, data_map
from plot import TIME_STEP_SEC

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

def find_sleep_probabilities(prob_mat, time):
    # Get the probabilities from the row corresponding to the nearest time
    time_index = time / TIME_STEP_SEC
    return prob_mat[time_index]

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

    find_state_mat_probabilities("4314139")
    #find_state_mat_probabilities()

if __name__ == "__main__":
    main()
