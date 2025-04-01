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
    def __init__(self, person_id):
        self.person_id = person_id
        self.time = []
        self.data = []

# Change this value to wherever the data is saved
DEFAULT_DATA_ROOT_DIR=f"{os.getenv('USERPROFILE')}/Downloads/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"
type_to_dir_name = {
    DataType.MOTION : "motion",
    DataType.HEART_RATE : "heart_rate",
    DataType.STATE : "labels"
}

# Data map has keys "MOTION" and "HEART_RATE", and value of DataContainer
# data_map[key].time gives a 1D list of timestamps
# data_map[key].data gives a 1D list of data
# For MOTION data, each element in the list is a tuple for the accelerations measured in g (x, y, z)
# For HEART_RATE data,hear each element in the list is 
data_map = {DataType.MOTION.name: [], 
            DataType.HEART_RATE.name: [],
            DataType.STATE.name: []}

def extract_csv_data(file, person_id, data_type: DataType) -> DataContainer:
    container = DataContainer(person_id)
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
                        if ground_truth >= 0:
                            # Data uses -1 to represent unknown state
                            container.time.append(float(line[0]))
                            container.data.append(ground_truth)
                    case _:
                        continue
    return container

def append_container(data_root_dir, person_id, key, value):
    match key:
        case DataType.MOTION:
            file_name = f"{person_id}_acceleration.txt"
        case DataType.HEART_RATE:
            file_name = f"{person_id}_heartrate.txt"
        case DataType.STATE:
            file_name = f"{person_id}_labeled_sleep.txt"

    data_path = os.path.join(data_root_dir, value, file_name)
    container = extract_csv_data(data_path, person_id, key)

    if (not any(c.person_id == person_id for c in data_map[key.name])):
        # prevent addition of duplicates to the map
        data_map[key.name].append(container)

# The key is to parse
def parse_data_file(data_root_dir: str, person_id: str, type_key=None) -> None:
    if type_key is None:
        # Can just read the files directly instead of walking through the entire directory
        for key, value in type_to_dir_name.items():
            append_container(data_root_dir, person_id, key, value)
    else:
        value = type_to_dir_name[type_key]
        append_container(data_root_dir, person_id, type_key, value)

def parse_data_files(data_root_dir: str, person_ids: List[str] = None, key=None):
    if person_ids is None and key is not None:
        person_ids = []
        # Assume we are using every person
        data_path = os.path.join(data_root_dir, type_to_dir_name[key])
        for _, _, files in os.walk(data_path):
            for file in files:
                split_name = file.split("_")
                person_ids.append(split_name[0])
            
    for person_id in person_ids:
        parse_data_file(data_root_dir, person_id, key)
