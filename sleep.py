from filterpy import ExtendedKalmanFilter as EKF
import numpy as np

from datetime import  datetime
from xml.etree import ElementTree as ET
import os

# Change this value to wherever the data is saved
DEFAULT_XML_SLEEP_DATA_PATH = f"{os.environ["USERPROFILE"]}/Downloads/export.xml"
DEFAULT_SEARCH_YEAR = "2022"

NUM_STATES = 0  # FIX LATER
NUM_MEASUREMENTS = 0

## MEASUREMENTS
hrv_data = []

# Sleep data is in minutes
sleep_data_dict = dict.fromkeys(["HKCategoryValueSleepAnalysisAwake", "HKCategoryValueSleepAnalysisAsleepCore", 
                                "HKCategoryValueSleepAnalysisAsleepDeep", "HKCategoryValueSleepAnalysisAsleepREM"], [])

def extract_hrv_data(record) -> None:
    hrv_metadata_list = record.find("HeartRateVariabilityMetadataList")
    # ibpm = InstantaneousBeatsPerMinute
    for ibpm_entry in hrv_metadata_list.findall("InstantaneousBeatsPerMinute"):
        bpm_value = ibpm_entry.get("bpm")
        hrv_data.append(int(bpm_value))

def extract_sleep_state_duration_data(record, sleep_key: str) -> None:
    start_date = record.get("startDate")
    end_date = record.get("endDate")

    # Perfom datetime subtraction to get relative hour / minute
    fmt_str = "%Y-%m-%d %H:%M:%S %z"
    start_date_fmt = datetime.strptime(start_date, fmt_str)
    end_date_fmt = datetime.strptime(end_date, fmt_str)

    duration = end_date_fmt - start_date_fmt
    total_mins = duration.total_seconds() / 60.0
    sleep_data_dict[sleep_key].append(total_mins)

def extract_all_from_xml_file(file_path: str) -> None:
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Parse each 'Record' tag
    for record in root.iter("Record"):
        source_name = record.get("sourceName")
        if source_name == "Jason's Apple Watch":
            record_type = record.get("type")
            match record_type:
                case "HKCategoryTypeIdentifierSleepAnalysis":
                    # Sleep analysis data
                    creation_date = record.get("creationDate")
                    if (creation_date.split("-")[0] == DEFAULT_SEARCH_YEAR):
                        # example date: "2025-03-26 23:23:39 -0400"
                        # split will give ['2025', '03', '26 23:23:39 ', ' 0400']
                        # so we get the first one in the list
                        extract_sleep_state_duration_data(record, record.get("value"))
                case "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":
                    # Heart rate variability data
                    extract_hrv_data(record)
                case _:
                    pass          

def HJacobian(x, *args):
    return

def Hx(x, *args):
    return


def main():
    # Extract the sleep measurements from the XML file
    extract_all_from_xml_file(DEFAULT_XML_SLEEP_DATA_PATH)

    # Initialize EKF
    Q_mat = ...
    R_mat = ...
    
    ekf = EKF(NUM_STATES, NUM_MEASUREMENTS)
    ekf.x = ... # initial state
    
    ekf.P = Q_mat
    ekf.R = R_mat
    ekf.Q = Q_mat
    ekf.F = ... # model
    ekf.H = None
    
    # Main Loop
    estimated_states = []
    i = 0
    z = ... # measurement readings, shape: [NUM_MEASUREMENTS, N]
    
    while True:
        curr_z = z[:, i]
        curr_z = z.reshape(-1, 1)
        
        ekf.predict_update(curr_z, HJacobian, Hx)
        posterior_state = ekf.x
        estimated_states.append(posterior_state)
        i+=1
    

if __name__ == "__main__":
    main() 

