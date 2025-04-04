import googlemaps
from datetime import datetime
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


# >> Google maps API

# Example code: https://github.com/edgargutgzz/sanpedro_trafficdata/blob/master/traffic.py#L24
# Documentation: https://developers.google.com/maps/documentation/distance-matrix/distance-matrix?_gl=1*t9cw1u*_up*MQ..*_ga*NDg1MTAzNzc2LjE3NDM2NDgzOTQ.*_ga_NRWSTWS78N*MTc0MzY0ODM5NC4xLjEuMTc0MzY0ODYzOS4wLjAuMA..#distance-matrix-advanced
API_KEY = 'AIzaSyCVFGP1yOSeNCDNFaQbx4EVYt3tKr_Z-Mc'
gmaps = googlemaps.Client(key=API_KEY)

HOME_ADDRESS_GPS = (43.4288496, -80.4105815)  # 208 Grand River Blvd, Kitchener, ON N2A 3G6
DESTINATION_GPS = (43.4692846, -80.5401371) # W Store, 200 University Avenue West, Waterloo, ON N2L 3G1
TARGET_TIME_OF_ARRIVAL = 8*60*60 # 9:00 AM in seconds

PRODUCTION = False

def get_traffic_duration_now(origin=HOME_ADDRESS_GPS, destination=DESTINATION_GPS):
    travel_time_with_traffic = 0

    if PRODUCTION:
        now = datetime.now()
        travel_time_data = gmaps.distance_matrix(  # API assumes travel via 'driving'
            origin,
            destination,
            departure_time=now
        )

        travel_time_with_traffic = travel_time_data['rows'][0]['elements'][0]['duration_in_traffic']['value']  # travel time in seconds (int)
    else:  # for dev
        # travel_time_with_traffic = 1036 # 17mins
        # travel_time_with_traffic = 1036
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
        self.time_to_leave = ctrl.Antecedent(np.arange(0, 121, 1), 'time_to_leave')  # time in minutes

        # Output variable
        self.alarm_pressure = ctrl.Consequent(np.arange(0, 101, 1), 'alarm_pressure')  # 5 parts

        # Membership functions
        self.wake_score['low'] = fuzz.trimf(self.wake_score.universe, [0, 0, 50])
        self.wake_score['medium'] = fuzz.trimf(self.wake_score.universe, [0, 50, 100])
        self.wake_score['high'] = fuzz.trimf(self.wake_score.universe, [50, 100, 100])

        self.time_to_leave.automf(3, names=['low','medium','high'])


        self.alarm_pressure.automf(5, names=['low', 'low-medium', 'medium', 'medium-high', 'high'])


        # Define rules
        # Option 1
        # self.rule1 = ctrl.Rule(self.time_to_leave['low'], self.alarm_pressure['high'])
        # self.rule2 = ctrl.Rule(self.wake_score['high'] & self.wake_score['medium'], self.alarm_pressure['medium-high'])
        # self.rule3 = ctrl.Rule(self.wake_score['low'] & self.time_to_leave['medium'], self.alarm_pressure['medium'])
        # self.rule4 = ctrl.Rule(self.time_to_leave['high'], self.alarm_pressure['low'])
        # self.alarm_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3, self.rule4])
        
        # Option 2
        self.rule1 = ctrl.Rule(self.wake_score['high'] | self.time_to_leave['low'], self.alarm_pressure['high'])
        self.rule2 = ctrl.Rule(self.wake_score['medium'] | self.time_to_leave['medium'], self.alarm_pressure['medium'])
        self.rule3 = ctrl.Rule(self.wake_score['low'] | self.time_to_leave['high'], self.alarm_pressure['low'])
        self.alarm_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3])

        # Control system
        self.simulation = ctrl.ControlSystemSimulation(self.alarm_ctrl)


    def __init__old_(self):
        # Define universe of discourse
        self.wake_score = ctrl.Antecedent(np.arange(0, 101, 1), 'wake_score')
        self.traffic_delay = ctrl.Antecedent(np.arange(0, 121, 1), 'traffic_delay')  # time in minutes
        self.time_to_deadline = ctrl.Antecedent(np.arange(0, 181, 1), 'time_to_deadline')
        
        # Output variable
        self.alarm_pressure = ctrl.Consequent(np.arange(0, 101, 1), 'alarm_pressure')
        
        # Membership functions
        self.wake_score['low'] = fuzz.trimf(self.wake_score.universe, [0, 0, 50])
        self.wake_score['medium'] = fuzz.trimf(self.wake_score.universe, [0, 50, 100])
        self.wake_score['high'] = fuzz.trimf(self.wake_score.universe, [50, 100, 100])

        self.traffic_delay['low'] = fuzz.trimf(self.traffic_delay.universe, [0, 0, 30])
        self.traffic_delay['medium'] = fuzz.trimf(self.traffic_delay.universe, [0, 30, 60])
        self.traffic_delay['high'] = fuzz.trimf(self.traffic_delay.universe, [30, 60, 120])

        self.time_to_deadline['tight'] = fuzz.trimf(self.time_to_deadline.universe, [0, 0, 90])
        self.time_to_deadline['moderate'] = fuzz.trimf(self.time_to_deadline.universe, [0, 90, 130])
        self.time_to_deadline['relaxed'] = fuzz.trapmf(self.time_to_deadline.universe, [90, 150, 180, 180])

        self.alarm_pressure['low'] = fuzz.trimf(self.alarm_pressure.universe, [0, 0, 50])
        self.alarm_pressure['medium'] = fuzz.trimf(self.alarm_pressure.universe, [0, 50, 100])
        self.alarm_pressure['high'] = fuzz.trimf(self.alarm_pressure.universe, [50, 100, 100])

        # Define rules
        # self.rule1 = ctrl.Rule(self.time_to_deadline['tight'], self.alarm_pressure['high'])
        self.rule1 = ctrl.Rule(
            self.wake_score['high'] & (self.traffic_delay['medium'] | self.traffic_delay['high']) & (self.time_to_deadline['tight']),
            self.alarm_pressure['high']
        )
        self.rule2 = ctrl.Rule(
            self.wake_score['medium'] & self.traffic_delay['medium'] & self.time_to_deadline['moderate'],
            self.alarm_pressure['medium']
        )
        self.rule3 = ctrl.Rule(
            self.wake_score['low'] & self.traffic_delay['low'] & self.time_to_deadline['relaxed'],
            self.alarm_pressure['low']
        )
        self.rule4 = ctrl.Rule(
            self.time_to_deadline['tight'] & (self.traffic_delay['high'] | self.wake_score['low']),
            self.alarm_pressure['high']
        )
        self.rule5 = ctrl.Rule(
            self.time_to_deadline['relaxed'] & (self.traffic_delay['low'] | self.wake_score['low']),
            self.alarm_pressure['low']
        )
        
        # Control system
        self.alarm_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3, self.rule4, self.rule5])
        self.simulation = ctrl.ControlSystemSimulation(self.alarm_ctrl)
        
        
    def __init__old(self):
        # Membership variables
        #    Antecendent = input varaiable
        self.wake_score = ctrl.Antecedent(np.arange(0, 101, 1), 'wake_score')
        self.traffic_delay = ctrl.Antecedent(np.arange(0, 121, 1), 'traffic_delay')  # takes time in minutes
        self.time_to_deadline = ctrl.Antecedent(np.arange(0, 181, 1), 'time_to_deadline')

        #    Consequent = output variable
        self.alarm_pressure = ctrl.Consequent(np.arange(0, 101, 1), 'alarm_pressure')


        # Membership functions
        self.wake_score.automf(3, names=['low', 'medium', 'high'])  # automatically creates membership function (automf) with 3 fuzzy sets (3)
        self.traffic_delay.automf(3, names=['low', 'medium', 'high'])
        self.time_to_deadline.automf(3, names=['relaxed', 'moderate', 'tight'])
        self.alarm_pressure.automf(3, names=['low', 'medium', 'high'])


        # # Rules for output variable (alarm)
        # rule1 = ctrl.Rule(wake_score['high'] | traffic_delay['high'], alarm_pressure['high'])
        # rule2 = ctrl.Rule(wake_score['medium'] & traffic_delay['medium'], alarm_pressure['medium'])
        # rule3 = ctrl.Rule(wake_score['low'] & traffic_delay['low'], alarm_pressure['low'])
        rule1 = ctrl.Rule(
            self.wake_score['high'] & (self.traffic_delay['medium'] | self.traffic_delay['high']) & (self.time_to_deadline['moderate'] | self.time_to_deadline['tight']),
            self.alarm_pressure['high']
        )
        rule2 = ctrl.Rule(
            self.wake_score['medium'] & self.traffic_delay['medium'] & self.time_to_deadline['moderate'],
            self.alarm_pressure['medium']
        )
        rule3 = ctrl.Rule(
            self.wake_score['low'] & self.traffic_delay['low'] & self.time_to_deadline['relaxed'],
            self.alarm_pressure['low']
        )
        rule4 = ctrl.Rule(
            self.time_to_deadline['tight'],
            self.alarm_pressure['high']
        )
        alarm_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
        self.simulation = ctrl.ControlSystemSimulation(alarm_ctrl)


    def compute_alarm_pressure(self, wake_score, time_to_leave):
        self.simulation.input['wake_score'] = wake_score
        self.simulation.input['time_to_leave'] = time_to_leave
        
        self.simulation.compute()
        # print(self.simulation.output['alarm_pressure'])
        if len(self.simulation.output) > 0:
            return self.simulation.output['alarm_pressure']
        else:
            print(wake_score, time_to_leave)
            return 0


    def compute_alarm_pressure_old(self, wake_score, traffic_delay, time_to_deadline):
        self.simulation.input['wake_score'] = wake_score
        self.simulation.input['traffic_delay'] = traffic_delay
        # print(time_to_deadline)
        self.simulation.input['time_to_deadline'] = time_to_deadline

        self.simulation.compute()
        # print(self.simulation.output['alarm_pressure'])
        if len(self.simulation.output) > 0:
            return self.simulation.output['alarm_pressure']
        else:
            print(wake_score, traffic_delay, time_to_deadline)
            return 0

    def plot_fuzzy_sets(self):
            self.wake_score.view()
            self.time_to_leave.view()
            self.alarm_pressure.view()

            plt.show()



    def plot_fuzzy_sets_old(self):
        self.wake_score.view()
        self.traffic_delay.view()
        self.time_to_deadline.view()
        self.alarm_pressure.view()
        
        plt.show()




# import numpy as np
# import skfuzzy as fuzz
# import skfuzzy.control as ctrl

# # Define fuzzy variables
# wake_score = ctrl.Antecedent(np.arange(0, 101, 1), 'wake_score')  # 0 to 100 scale
# traffic_urgency = ctrl.Antecedent(np.arange(0, 101, 1), 'traffic_urgency')
# alarm_pressure = ctrl.Consequent(np.arange(0, 101, 1), 'alarm_pressure')

# # Define fuzzy membership functions
# wake_score['low'] = fuzz.trimf(wake_score.universe, [0, 0, 50])
# wake_score['medium'] = fuzz.trimf(wake_score.universe, [30, 50, 70])
# wake_score['high'] = fuzz.trimf(wake_score.universe, [50, 100, 100])

# traffic_urgency['low'] = fuzz.trimf(traffic_urgency.universe, [0, 0, 50])
# traffic_urgency['moderate'] = fuzz.trimf(traffic_urgency.universe, [30, 50, 70])
# traffic_urgency['high'] = fuzz.trimf(traffic_urgency.universe, [50, 100, 100])

# alarm_pressure['low'] = fuzz.trimf(alarm_pressure.universe, [0, 0, 50])
# alarm_pressure['medium'] = fuzz.trimf(alarm_pressure.universe, [30, 50, 70])
# alarm_pressure['high'] = fuzz.trimf(alarm_pressure.universe, [50, 100, 100])

# # Define fuzzy rules
# rule1 = ctrl.Rule(wake_score['low'] & traffic_urgency['high'], alarm_pressure['high'])
# rule2 = ctrl.Rule(wake_score['medium'] & traffic_urgency['moderate'], alarm_pressure['medium'])
# rule3 = ctrl.Rule(wake_score['high'] & traffic_urgency['low'], alarm_pressure['low'])
# rule4 = ctrl.Rule(wake_score['high'] & traffic_urgency['high'], alarm_pressure['high'])

# # Create control system
# alarm_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
# alarm_sim = ctrl.ControlSystemSimulation(alarm_ctrl)

# # Example input values
# alarm_sim.input['wake_score'] = 40  # Example wake score
# alarm_sim.input['traffic_urgency'] = 80  # Example traffic estimate

# # Compute output
# alarm_sim.compute()
# print("Alarm Pressure:", alarm_sim.output['alarm_pressure'])

# # Trigger alarm if above threshold
# alarm_threshold = 70
# if alarm_sim.output['alarm_pressure'] >= alarm_threshold:
#     print("Trigger Alarm!")