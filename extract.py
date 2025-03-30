from enum import Enum
from typing import List
import csv
import os

## MEASUREMENTS
class DataType(Enum):
    MOTION = 1,
    HEART_RATE = 2,
    STATE = 3,  # ground truth data

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
