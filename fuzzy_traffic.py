import googlemaps
from datetime import datetime
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# >> Google maps API

# Example code: https://github.com/edgargutgzz/sanpedro_trafficdata/blob/master/traffic.py#L24
# Documentation: https://developers.google.com/maps/documentation/distance-matrix/distance-matrix?_gl=1*t9cw1u*_up*MQ..*_ga*NDg1MTAzNzc2LjE3NDM2NDgzOTQ.*_ga_NRWSTWS78N*MTc0MzY0ODM5NC4xLjEuMTc0MzY0ODYzOS4wLjAuMA..#distance-matrix-advanced
API_KEY = 'AIzaSyCVFGP1yOSeNCDNFaQbx4EVYt3tKr_Z-Mc'
gmaps = googlemaps.Client(key=API_KEY)

HOME_ADDRESS_GPS = (43.4288496, -80.4105815)  # 208 Grand River Blvd, Kitchener, ON N2A 3G6
DESTINATION_GPS = (43.4692846, -80.5401371) # W Store, 200 University Avenue West, Waterloo, ON N2L 3G1

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
        travel_time_with_traffic = 5036

    return travel_time_with_traffic


class FuzzyAlarm():
    def __init__(self):
        # Membership variables
        #    Antecendent = input varaiable
        wake_score = ctrl.Antecedent(np.arange(0, 101, 1), 'wake_score')
        traffic_delay = ctrl.Antecedent(np.arange(0, 121, 1), 'traffic_delay')  # takes time in minutes

        #    Consequent = output variable
        alarm_pressure = ctrl.Consequent(np.arange(0, 101, 1), 'alarm_pressure')


        # Membership functions
        wake_score.automf(3, names=['low', 'medium', 'high'])  # automatically creates membership function (automf) with 3 fuzzy sets (3)
        traffic_delay.automf(3, names=['low', 'medium', 'high'])
        alarm_pressure.automf(3, names=['low', 'medium', 'high'])


        # Rules for output variable (alarm)
        rule1 = ctrl.Rule(wake_score['high'] | traffic_delay['high'], alarm_pressure['high'])
        rule2 = ctrl.Rule(wake_score['medium'] & traffic_delay['medium'], alarm_pressure['medium'])
        rule3 = ctrl.Rule(wake_score['low'] & traffic_delay['low'], alarm_pressure['low'])
        alarm_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        self.simulation = ctrl.ControlSystemSimulation(alarm_ctrl)


    def compute_alarm_pressure(self, wake_score, traffic_delay):
        self.simulation.input['wake_score'] = wake_score
        self.simulation.input['traffic_delay'] = traffic_delay

        self.simulation.compute()
        return self.simulation.output['alarm_pressure']





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