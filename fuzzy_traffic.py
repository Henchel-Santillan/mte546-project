import googlemaps
from datetime import datetime
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


# >> Google maps API

# Example code: https://github.com/edgargutgzz/sanpedro_trafficdata/blob/master/traffic.py#L24
# Documentation: https://developers.google.com/maps/documentation/distance-matrix/distance-matrix?_gl=1*t9cw1u*_up*MQ..*_ga*NDg1MTAzNzc2LjE3NDM2NDgzOTQ.*_ga_NRWSTWS78N*MTc0MzY0ODM5NC4xLjEuMTc0MzY0ODYzOS4wLjAuMA..#distance-matrix-advanced
API_KEY = ''
gmaps = None

HOME_ADDRESS_GPS = (43.4288496, -80.4105815)  # 208 Grand River Blvd, Kitchener, ON N2A 3G6
DESTINATION_GPS = (43.4692846, -80.5401371) # W Store, 200 University Avenue West, Waterloo, ON N2L 3G1
TARGET_TIME_OF_ARRIVAL = 8*60*60 # 9:00 AM in seconds

PRODUCTION = False

def get_traffic_duration_now(origin=HOME_ADDRESS_GPS, destination=DESTINATION_GPS):
    travel_time_with_traffic = 0

    if PRODUCTION:
        gmaps = googlemaps.Client(key=API_KEY)
        now = datetime.now()
        travel_time_data = gmaps.distance_matrix(  # API assumes travel via 'driving'
            origin,
            destination,
            departure_time=now
        )

        travel_time_with_traffic = travel_time_data['rows'][0]['elements'][0]['duration_in_traffic']['value']  # travel time in seconds (int)
    else:  # for dev
        travel_time_with_traffic = 3600 # one hour commute in seconds

    return travel_time_with_traffic


def get_time_to_deadline(time_elapsed: int=None):
    current_time_in_seconds = 0  # if used (dev), it assumes the person begins sleep at midnight
    if PRODUCTION:
        now = datetime.now()
        current_time_in_seconds = now.hour * 3600 + now.minute * 60 + now.second
    else:
        current_time_in_seconds += time_elapsed
    time_remaining = (TARGET_TIME_OF_ARRIVAL - current_time_in_seconds) // 60  # convert to minutes
    return max(0, time_remaining)  # Ensure no negative values


class FuzzyAlarm():
    def __init__(self):
        # Define universe of discourse
        self.wake_score = ctrl.Antecedent(np.arange(0, 101, 1), 'wake_score')
        self.time_to_leave = ctrl.Antecedent(np.arange(0, 120, 1), 'time_to_leave')  # time in minutes

        # Output variable
        self.alarm_pressure = ctrl.Consequent(np.arange(0, 101, 1), 'alarm_pressure')  # 5 parts

        # Membership functions
        self.wake_score['low'] = fuzz.trimf(self.wake_score.universe, [0, 0, 50])
        self.wake_score['medium'] = fuzz.trimf(self.wake_score.universe, [0, 50, 100])
        self.wake_score['high'] = fuzz.trimf(self.wake_score.universe, [50, 100, 100])

        self.time_to_leave['tight'] = fuzz.trapmf(self.time_to_leave.universe, [0, 0, 10, 30])
        self.time_to_leave['moderate'] = fuzz.trapmf(self.time_to_leave.universe, [10, 30, 38, 70])
        self.time_to_leave['relaxed'] = fuzz.trapmf(self.time_to_leave.universe, [35, 80, 120, 120])

        self.alarm_pressure.automf(4, names=['low', 'medium', 'medium-high', 'high'])

        # Define rules
        self.rule1 = ctrl.Rule(self.time_to_leave['tight'], self.alarm_pressure['high'])
        self.rule2 = ctrl.Rule(self.wake_score['high'] & self.time_to_leave['moderate'], self.alarm_pressure['medium-high'])
        self.rule3 = ctrl.Rule(self.wake_score['low'] & self.time_to_leave['moderate'], self.alarm_pressure['medium'])
        self.rule4 = ctrl.Rule(self.time_to_leave['relaxed'], self.alarm_pressure['low'])
        self.alarm_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3, self.rule4])

        # Control system
        self.simulation = ctrl.ControlSystemSimulation(self.alarm_ctrl, lenient=False)


    def compute_alarm_pressure(self, wake_score, time_to_leave):
        self.simulation.input['wake_score'] = wake_score
        self.simulation.input['time_to_leave'] = time_to_leave
        
        self.simulation.compute()

        if len(self.simulation.output) > 0:
            return self.simulation.output['alarm_pressure']
        else:
            print(wake_score, time_to_leave)
            return 0


    def plot_fuzzy_sets(self):
            self.wake_score.view()
            self.time_to_leave.view()
            self.alarm_pressure.view()

            plt.show()
