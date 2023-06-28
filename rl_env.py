import cityflow
import os
import json
import torch
import gym
import numpy as np
import random

config_path = 'configs/config.json'

class Map:
    def __init__(self, roadnet_path, sensor_length) -> None:
        self.intersection_dict = {}
        self.intersections = []
        self.roads = []
        self.road_dict = {}
        self.sensor_length = sensor_length
        with open(roadnet_path, 'r') as f:
            self.roadnet = json.load(f)
        self.initialize()
    
    def initialize(self):
        # self.intersections is all the intersections that are not virtual (i.e. virtual = at the edges of the map)
        self.intersections = [i for i in self.roadnet['intersections'] \
            if not i['virtual']]
        self.roads = self.roadnet['roads']
        # Set up self.intersections which is a list of each of the intersections
        for i in self.intersections:
            id = i['id']
            self.intersection_dict[id] = i
            self.intersection_dict[id]['incoming_roads'] = \
                list(set([temp['startRoad'] for temp in i['roadLinks']]))
            self.intersection_dict[id]['outgoing_roads'] = \
                list(set([temp['endRoad'] for temp in i['roadLinks']]))
            # Example of what phases could look like:
            # {0: [0, 2, 3, 6, 7, 10], 1: [10, 2, 3, 6], 2: [1, 2, 3, 6, 8, 10], 3: [10, 2, 3, 6], 4: [2, 3, 4, 6, 10, 11], 5: [10, 2, 3, 6], 6: [2, 3, 5, 6, 9, 10], 7: [10, 2, 3, 6]}
            self.intersection_dict[id]['phases'] = {
                phase: x['availableRoadLinks'] for phase, x in enumerate(i['trafficLight']['lightphases'])
                }
            
            keys = ['type', 'startRoad', 'endRoad', 'direction']
            self.intersection_dict[id]['phases_links'] = {}
            self.intersection_dict[id]["phase_incoming_lanes"] = {}
            for phase, x in enumerate(i['trafficLight']['lightphases']):
                phase_incoming_lanes = set()
                roadlinks = []
                for roadlink_index in x['availableRoadLinks']:
                    roadlink = i['roadLinks'][roadlink_index]
                    for lane_link in roadlink["laneLinks"]:
                        phase_incoming_lanes.add(f"{roadlink['startRoad']}_{lane_link['startLaneIndex']}")
                    roadlinks.append({k: v for k, v in roadlink.items() if k in keys})
                self.intersection_dict[id]["phase_incoming_lanes"][phase] = list(phase_incoming_lanes)
                self.intersection_dict[id]['phases_links'] = roadlinks
                    
        # What is movements?
        self.initialize_movements()

        # Initialize road dict in the same way as intersection dict
        # Key is the road id, value is the original road, we add 'lane_ids' as a 
        #   key of the original road and define them to be the "{road id}_{index of lane}" so each lane as a unique id
        for r in self.roads:
            id = r['id']
            self.road_dict[id] = r
            self.road_dict[id]['lane_ids'] = [f"{id}_{i}" for i in range(len(r['lanes']))]

        self.initialize_lane_dict()
    
    def get_intersections(self):
        return self.intersections
    
    def get_intersection_info(self, intersection_id):
        return self.intersection_dict[intersection_id]

    def get_intersection_incoming_roads(self, intersection_id):
        return self.intersection_dict[intersection_id]['incoming_roads']
    
    def get_intersection_outgoing_roads(self, intersection_id):
        return self.intersection_dict[intersection_id]['outgoing_roads']  
    
    def get_intersection_phases(self, intersection_id):
        return self.intersection_dict[intersection_id]['phases']

    def get_intersection_phase_roadlinks(self, intersection_id, phase):
        return self.intersection_dict[intersection_id]['phases_links'][phase]
    
    def get_road_lanes(self, road_id):
        return self.road_dict[road_id]['lane_ids']
    
    def get_intersection_incoming_lanes(self, intersection_id):
        incoming_roads = self.get_intersection_incoming_roads(intersection_id)
        incoming_lanes = []
        for r in incoming_roads:
            incoming_lanes.extend(self.get_road_lanes(r))
        return list(set(incoming_lanes))
    
    def initialize_movements(self): 
        # Movements are used by the maximum pressure reward function, see PressLight paper for definition.
        for intersection in self.get_intersections():
            roadlinks = self.get_intersection_info(intersection['id'])['roadLinks']
            movements = {}
            for i, roadlink in enumerate(roadlinks):
                start_lanes_indices = list(set([l['startLaneIndex'] for l in roadlink['laneLinks']]))
                start_lane_ids = [f"{roadlink['startRoad']}_{i}" for i in start_lanes_indices]
                end_lanes_indices = list(set([l['endLaneIndex'] for l in roadlink['laneLinks']]))
                end_lane_ids = [f"{roadlink['endRoad']}_{i}" for i in end_lanes_indices]
                movement = {'incoming_lanes': start_lane_ids, 'outgoing_lanes': end_lane_ids, 'type': roadlink['type']}
                movements[i] = movement
            intersection['movements'] = movements
    
    def get_intersection_movements(self, intersection_id):
        return self.get_intersection_info(intersection_id)['movements']
    
    def initialize_lane_dict(self):
        lane_dict = {}
        for road in self.roads:
            for lane_id in road['lane_ids']:
                lane_dict[lane_id] = {}

        for intersection in self.intersection_dict:
            for road_link in self.get_intersection_info(intersection)['roadLinks']:
                start_road = road_link['startRoad']
                end_road = road_link['endRoad']
                for lane_link in road_link['laneLinks']:
                    start_lane = f"{start_road}_{lane_link['startLaneIndex']}"
                    end_lane = f"{end_road}_{lane_link['endLaneIndex']}"
                    # Does this miss setting some starts and ends to lanes if the lane ends at a virtual intersection?
                    # Yes, but we set this in the for loop below! We go through each road without a non-virtual intersection at its end
                    #   and set the lanes start and ends in it accordingly

                    lane_dict[start_lane]['end'] = np.array([lane_link['points'][0]['x'], lane_link['points'][0]['y']])
                    lane_dict[end_lane]['start'] = np.array([lane_link['points'][-1]['x'], lane_link['points'][-1]['y']])
                    
        for road in self.roads:
            points = road['points']
            start = np.array([points[0]['x'], points[0]['y']])
            end = np.array([points[1]['x'], points[1]['y']])
            direction = end - start
            perp = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
            if road['startIntersection'] not in self.intersection_dict:
                for lane_id, lane_info in zip(road['lane_ids'], road['lanes']):
                    lane_index = int(lane_id.split("_")[-1])
                    width = lane_info['width']
                    lane_dict[lane_id]['start'] = start - perp * (lane_index + 0.5) * width
            
            if road['endIntersection'] not in self.intersection_dict:
                for lane_id, lane_info in zip(road['lane_ids'], road['lanes']):
                    lane_index = int(lane_id.split("_")[-1])
                    width = lane_info['width']
                    lane_dict[lane_id]['end'] = end - perp * (lane_index + 0.5) * width

        for lane in lane_dict:
            lane_dict[lane]['distance'] = np.linalg.norm(lane_dict[lane]['end'] - lane_dict[lane]['start'])
        
        for lane in lane_dict:
            lane_dict[lane]['sensorStart'] = lane_dict[lane]['distance'] - 50
            lane_dict[lane]['sensorEnd'] = lane_dict[lane]['sensorStart'] + self.sensor_length

        self.lane_dict = lane_dict
    
    def get_lane_loc_info(self, lane_id):
        return self.lane_dict[lane_id]

class CustomEngine(cityflow.Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entrance_times = {}
        self.exit_times = {}
        self.current_vehicles = set()
    
    def update_times(self):
        curr_vehicles_ids = set(self.get_vehicles(include_waiting=True))
        for id in curr_vehicles_ids:
            if id not in self.entrance_times:
                self.entrance_times[id] = self.get_current_time()
        for id in self.current_vehicles:
            if id not in curr_vehicles_ids and id not in self.exit_times:
                self.exit_times[id] = self.get_current_time()
        self.current_vehicles = curr_vehicles_ids
    
    def next_step(self):
        self.update_times()
        super().next_step()

class TrafficEnvironment(gym.Env):
    min_phase_length = 5.0
    action_interval = 1.0
    time_per_episode = 500.0

    def __init__(self, history_size, sensor_based, config_path, thread_num=1):
        super().__init__()
        self.eng = CustomEngine(config_path, thread_num=thread_num)
        with open(config_path, "r") as f:
            self.configs = json.load(f)
        self.map = Map(os.path.join(self.configs['dir'], self.configs['roadnetFile']), 10) 
        self.eng_time_step = self.configs['interval']
        self.single_intersection = len(self.map.get_intersections()) == 1
        self.num_intersections = len(self.map.get_intersections())
        self.sensor_based = sensor_based

        self.history_size = history_size
        self.snapshot_size = -1
        self.average_travel_times = []
        self.rewards = []
        self.total_reward = 0
        self.current_phase = {}

        self._init_history()
        for intersection in self.map.get_intersections():

            self.current_phase[intersection['id']] = {"time_of_change": 0.0, "current_phase": 0}
        self.action_space = gym.spaces.Discrete(4) # Hard coded, fix later! # gym.spaces.MultiDiscrete(num_actions)
        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf, shape=self._get_obs().shape) # gym.spaces.Dict(obs_dict)

    def _sample_controlling_intersection(self):
        '''
        Samples an intersection to be controlled by the RL agent in each training run. Ignored during inference time.
        '''
        total_intersections = len(self.map.get_intersections())
        random_index = random.randrange(0, total_intersections)
        self.controlling_intersection = self.map.get_intersections()[random_index]['id']
        print(f"Controlling intersection {self.controlling_intersection} with RL.")

    def _engine_step(self, step):
        time_passed = 0
        while time_passed < step:
            self.eng.next_step()
            self._run_traffic_light_algo()
            self.update_history()
            time_passed += self.eng_time_step
    
    def _init_history(self):
        '''
        Initializes the data space to be all zeros initially and builds it from there.
        '''
        self.snapshot_size = []
        self.history = {}
        for intersection in self.map.get_intersections():
            for phase in [0, 2, 4, 6]:
                num_incoming_lanes_phase = len(intersection["phase_incoming_lanes"][phase])
                break
            self.snapshot_size = [4, num_incoming_lanes_phase + 1, self.history_size]
            assert self.snapshot_size == [4, 7, self.history_size]
            self.history[intersection['id']] = np.zeros(self.snapshot_size)
        
        self._sample_controlling_intersection() # Select an intersection to control for the first run

    def reset(self):
        '''
        Resets the environment to time zero with no cars.
        '''
        self.rewards.append(self.total_reward)
        self.total_reward = 0
        self.average_travel_times.append(self.eng.get_average_travel_time())
        self._init_history()
        self.eng.reset()
        for intersection in self.map.get_intersections():
            self.current_phase[intersection['id']] = {"time_of_change": 0.0, "current_phase": 0} # Time
        return self._get_obs()

    def _set_tl_phase(self, intersection_id, action):
        # print(f"Setting light {intersection_id} to phase {action}.")
        # This calls self.eng.set_tl_phase and also sets current_phase correctly,
        # self.eng.set_tl_phase should not be called anywhere else.
        self.eng.set_tl_phase(intersection_id, action)
        self.current_phase[intersection_id]['current_phase'] = action
        self.current_phase[intersection_id]['time_of_change'] = self.eng.get_current_time()

    
    def step(self, action):
        info = {}

        # check if action allowed, if not set info to -1 for this intersection
        # action is allowed if: previous phase change is >= 5 seconds ago
        # assert len(action) == len(self.map.get_intersections())
        curr_time = self.eng.get_current_time()
        if curr_time - self.current_phase[self.controlling_intersection]['time_of_change'] < self.min_phase_length:
            info[self.controlling_intersection] = -1
        else:
            info[self.controlling_intersection] = 1
            if action != self.current_phase[self.controlling_intersection]['current_phase']:
                self._set_tl_phase(self.controlling_intersection, action)
                
        self._engine_step(self.action_interval)

        state = self._get_obs()
        # Why do we sum the reward across all intersections at once?
        reward = self._get_max_pressure()[self.controlling_intersection] # TODO: Implement other reward functions
        done = self.eng.get_current_time() >= self.time_per_episode
        
        self.total_reward += reward
        return state, reward, done, info
    
    def _get_curr_snapshot(self):
        if self.sensor_based:
            vehicle_counts = self._get_lane_sensor_values()
        else:
            vehicle_counts = self.eng.get_lane_vehicle_count()
        # if self.single_intersection:
        #     intersection = self.map.get_intersections()[0]
        #     incoming_lanes = self.map.get_intersection_incoming_lanes(intersection['id'])
        #     obs = np.zeros(len(incoming_lanes) + 1)
        #     for i, lane in enumerate(incoming_lanes):
        #         obs[i] = vehicle_counts[lane]
        #     obs[-1] = self.current_phase
        #     return obs
        # else:
        snapshot = {}
        for intersection in self.map.get_intersections():
            new_snapshot = np.zeros((4, 7, 1)) # 4 phases, 6 lanes per phase, 1 history

            intersection_snapshot = []
            for phase in [0, 2, 4, 6]:
                incoming_lanes = intersection["phase_incoming_lanes"][phase]
                data = [vehicle_counts[lane] for lane in incoming_lanes] + [ self.current_phase[intersection['id']]['current_phase'] == phase ]
                assert len(data) == 7
                intersection_snapshot.append(data)
            intersection_snapshot = np.array(intersection_snapshot)

            snapshot[intersection['id']] = intersection_snapshot
        return snapshot
    
    def update_history(self):
        snapshot = self._get_curr_snapshot()
        
        for intersection_id in self.history:
            original_size = self.history[intersection_id].shape
            self.history[intersection_id] = np.concatenate([self.history[intersection_id][:, :, 1:], \
                np.expand_dims(snapshot[intersection_id], axis=-1)], axis=-1)
            assert original_size == self.history[intersection_id].shape

    def _get_obs(self):
        return self.history[self.controlling_intersection] # Only return data for the intersection being controlled

    def _get_max_pressure(self):
        '''
        Compute the maximum pressure among all intersections in the simulation.

        Returns:
            If the simulation is set up to only simulate one intersection, returns the negative absolute pressure for that intersection as a float.
            Otherwise, returns a dictionary where the keys are the IDs of the intersections and the values are the negative absolute pressures for the intersections as floats.
        '''
        neg_abs_pressures = {}
        vehicle_counts = self.eng.get_lane_vehicle_count()
        for intersection in self.map.get_intersections():
            movements = self.map.get_intersection_movements(intersection['id']).values()
            pressure = 0
            for movement in movements:
                pressure += sum([vehicle_counts[lane] for lane in movement['incoming_lanes']])
                pressure -= sum([vehicle_counts[lane] for lane in movement['outgoing_lanes']])
                # TODO: Implement penalty for overflowing capacity
            neg_abs_pressures[intersection['id']] = -abs(pressure)

        if self.single_intersection:
            return neg_abs_pressures[0]
        else:
            return neg_abs_pressures
    
    def _get_lane_sensor_values(self):
        vehicle_distances = self.eng.get_vehicle_distance()
        vehicle_counts = {}
        for lane, vehicles in self.eng.get_lane_vehicles().items():
            vehicle_counts[lane] = 0
            lane_info = self.map.get_lane_loc_info(lane)
            for vehicle in vehicles:
                if lane_info['sensorStart'] < vehicle_distances[vehicle] < lane_info['sensorEnd']:
                    vehicle_counts[lane] = 1
                    break
        return vehicle_counts


    def _run_traffic_light_algo(self):
        '''
        Control all the other lights not being controlled by the RL agent. This 
        eventually can use the RL agent to still control the other intersections if we 
        save the model as a checkpoint and reference it. For now I will use a simple fixed timing algorithm.
        '''
        current_time = self.eng.get_current_time()
        for intersection in self.map.get_intersections():
            # Avoid controlling the controlling intersection, let the RL agent control that
            id = intersection["id"]
            if id == self.controlling_intersection:
                continue 
            
            phase = self.current_phase[id]["current_phase"]
            if current_time - self.current_phase[id]["time_of_change"] >= 20:
                phase += 2
                if phase < 0 or phase > 6:
                    phase = 0
                self._set_tl_phase(id, phase)
            
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
