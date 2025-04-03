import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from enum import Enum

from extract import data_map, DataType

class ImuAxis(Enum):
    X_AXIS = 1
    Y_AXIS = 2
    Z_AXIS = 3

TIME_STEP_SEC = 30  # time step our simulation loops through (in seconds)

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

def plot_motion_data(person_i: int, should_print=False) -> None:
    """
    MODELS:
    ax: -7.124e-17x^4 + 4.18e-12x^3 - 7.583e-08x^2 + 0.0004x + 1.683e-07
    ay: -5.469e-17x^4 + 3.283e-12x^3 - 5.88e-08x^2 + 0.0003x + 1.205e-07
    az: 1.011e-16x^4 - 5.957e-12x^3 + 1.084e-07  - 0.0006x - 2.473e-07
    """
    def gen_sensor_model_motion(person_i: int, axis: ImuAxis, should_print=should_print):
        motion_time = data_map[DataType.MOTION.name][person_i].time
        motion_val = data_map[DataType.MOTION.name][person_i].data

        # Regressors / intercepts
        X = np.column_stack([np.ones_like(motion_time), 
                            motion_time, 
                            [time ** 2 for time in motion_time],
                            [time ** 3 for time in motion_time],
                            [time ** 4 for time in motion_time]])

        # Fit robust model (Tukey bisquare)
        index = axis.value - 1

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
    # for state_i in range(len(labels)):
    #     if labels[state_i] != "Wake Score":
    #         axes[0].plot(time_range[: len(time_range)-1], states[state_i, :], label=labels[state_i])

    for state_i in range(0, 3 + 1):
        axes[0].plot(time_range[: len(time_range)-1], - 1 * states[state_i, :], label=labels[state_i])

    for state_i in range(5, len(labels)):
        axes[0].plot(time_range[: len(time_range)-1], states[state_i, :], label=labels[state_i])

    axes[0].set_title("Predicted States")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("State")
    axes[0].legend()
    axes[0].grid(True)

    # Ground truth state vs predicted state (high %)
    transformed_groundtruth_states = np.where(ground_truth_states == 2, 1, ground_truth_states)
    transformed_groundtruth_states = np.where(transformed_groundtruth_states == 3, 2, transformed_groundtruth_states)
    transformed_groundtruth_states = np.where(transformed_groundtruth_states == 5, 3, transformed_groundtruth_states)
    axes[1].plot(time_range[: len(time_range)-1], transformed_groundtruth_states, label="Ground Truth sleep state")  # convert labelled y-axis to values that are easier to comprehend

    indices_of_highest_probability_states = np.argmax(states[:4, :], axis=0)  # 1xN (0: awake, 1: core, 2: deep, 3: rem)
    axes[1].plot(time_range[: len(time_range)-1], indices_of_highest_probability_states, label="Predicted Sleep States")

    state_labels = {0: "Awake", 1: "Core", 2: "Deep", 3: "REM"}  # change y-axis labels for clarity
    axes[1].set_yticks(list(state_labels.keys()))
    axes[1].set_yticklabels(list(state_labels.values()))

    axes[1].set_title("Final Predicted Sleep State")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("State")
    axes[1].legend()
    axes[1].grid(True)

    # Sleep score
    axes[2].plot(time_range[: len(time_range)-1], states[4, :])
    axes[2].set_title("Sleep Score")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Sleep Score")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_ekf_states_and_alarm(states: np.array, ground_truth_states: list, alarms_vals: np.array, start_time: float, final_time: float):
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

    fig, axes = plt.subplots(4, 1, figsize=(16,8))

    # EKF outputs
    # for state_i in range(len(labels)):
    #     if labels[state_i] != "Wake Score":
    #         axes[0].plot(time_range[: len(time_range)-1], states[state_i, :], label=labels[state_i])

    for state_i in range(0, 3 + 1):
        axes[0].plot(time_range[: len(time_range)-1], - 1 * states[state_i, :], label=labels[state_i])

    for state_i in range(5, len(labels)):
        axes[0].plot(time_range[: len(time_range)-1], states[state_i, :], label=labels[state_i])

    axes[0].set_title("Predicted States")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("State")
    axes[0].legend()
    axes[0].grid(True)

    # Ground truth state vs predicted state (high %)
    transformed_groundtruth_states = np.where(ground_truth_states == 2, 1, ground_truth_states)
    transformed_groundtruth_states = np.where(transformed_groundtruth_states == 3, 2, transformed_groundtruth_states)
    transformed_groundtruth_states = np.where(transformed_groundtruth_states == 5, 3, transformed_groundtruth_states)
    axes[1].plot(time_range[: len(time_range)-1], transformed_groundtruth_states, label="Ground Truth sleep state")  # convert labelled y-axis to values that are easier to comprehend

    indices_of_highest_probability_states = np.argmax(states[:4, :], axis=0)  # 1xN (0: awake, 1: core, 2: deep, 3: rem)
    axes[1].plot(time_range[: len(time_range)-1], indices_of_highest_probability_states, label="Predicted Sleep States")

    state_labels = {0: "Awake", 1: "Core", 2: "Deep", 3: "REM"}  # change y-axis labels for clarity
    axes[1].set_yticks(list(state_labels.keys()))
    axes[1].set_yticklabels(list(state_labels.values()))

    axes[1].set_title("Final Predicted Sleep State")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("State")
    axes[1].legend()
    axes[1].grid(True)


    # Sleep score
    axes[2].plot(time_range[: len(time_range)-1], states[4, :])
    axes[2].set_title("Sleep Score")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Sleep Score")
    axes[2].grid(True)


    # Alarm Pressure
    axes[3].plot(time_range[: len(time_range)-1], alarms_vals[0, :])  # alarm value

    set_alarms = alarms_vals[1, :]
    set_alarms = np.array(set_alarms, dtype=bool)

    axes[3].fill_between(time_range[: len(time_range)-1], axes[3].get_ylim()[0], axes[3].get_ylim()[1], where=set_alarms,
                        color='lightgreen', alpha=0.3, label="Set off alarm")
    axes[3].fill_between(time_range[: len(time_range)-1], axes[3].get_ylim()[0], axes[3].get_ylim()[1], where=~set_alarms,
                        color='lightcoral', alpha=0.3, label="Set off alarm")
    axes[3].set_title("Alarm Pressure")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Alarm Pressure")
    axes[3].grid(True)


    plt.tight_layout()
    plt.show()

