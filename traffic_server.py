from concurrent import futures
import grpc
import time
import traffic_pb2
import traffic_pb2_grpc
import threading as th
import cityflow
import json
from tqdm import tqdm
import binascii
import os
import datetime

from prototype import TrafficEnvironment, CustomEngine
from stable_baselines3 import PPO
import numpy as np
        
INITIALIZE_WAIT_TIME=2
SERVER_REFRESH_TIME=0.125
ITERATIONS=1000
STANDARD_PHASE_TIME = 20
MINIMUM_PHASE_TIME = 2
MAXIMUM_PHASE_TIME = 40
USE_MODEL = False
USE_FIXED_TIME = False
CAR_DISCOUNT_FACTOR = 1.333
TOKEN_LENGTH = 15
RETRY_ATTEMPTS = 2


class TrafficServerServicer(traffic_pb2_grpc.TrafficServerServicer):
    def __init__(self, port, intersection_tokens, weighted_consensus=False) -> None:
        '''
        Initializes a new `TrafficServerServicer` object with the specified `port`, `intersection_tokens`, and `weighted_consensus` settings.

        Args:
            port (int): The port number used by the server.
            intersection_tokens (dict): A dictionary of intersection IDs and their corresponding API keys.
            weighted_consensus (bool): Whether to use weighted consensus when updating lane data. Defaults to False.
        Returns:
            None
        '''
        super().__init__()
        self.port = port
        self.clock = 0
        self.lane_data = {}
        self.phase_status = {}
        self.intersection_tokens = intersection_tokens
        self.last_updated = 0.0
        self.weighted = weighted_consensus
            
    def HeartBeat(self, request, context):
        '''
        Verifies the API key for a given intersection and updates the phase status for that intersection if necessary.

        Args:
            request (traffic_pb2.HeartBeatRequest): The request object containing the intersection ID and API key to verify, and the phase status to update.
            context (grpc._server._Context): The context of the current RPC request.
        Returns:
            A `traffic_pb2.RefreshReply` object containing the version number, current simulation time, and error code.
        '''
        if request.intersection_id and (request.intersection_id in self.intersection_tokens) and request.api_key and self.intersection_tokens[request.intersection_id] == request.api_key:
            pass
        else:
            return traffic_pb2.RefreshReply(version=1, time=self.clock, error_code="432")
        if request.intersection_id and request.phase_id:
            self.phase_status[request.intersection_id] = request.phase_id
        return traffic_pb2.RefreshReply(version=1, time=self.clock, error_code="200")

    def TransmitData(self, request, context):
        '''
        Receives and processes lane data updates for a given intersection.

        Args:
            request (traffic_pb2.TransmitDataRequest): The request object containing the intersection ID and API key to verify, and the lane data updates to process.
            context (grpc._server._Context): The context of the current RPC request.
        Returns:
            A `traffic_pb2.DataReply` object containing the version number, current simulation time, and error code.
        '''
        if request.intersection_id and (request.intersection_id in self.intersection_tokens) and request.api_key and self.intersection_tokens[request.intersection_id] == request.api_key:
            pass
        else:
            return traffic_pb2.DataReply(version=1, time=self.clock, error_code="432")
        
        if request.timestamp and self.last_updated < float(request.timestamp):
            self.last_updated = float(request.timestamp)
            if request.entries:
                for entry in request.entries:
                    self.lane_data[entry.key] = max(entry.value, 0)
        elif request.timestamp and self.last_updated == float(request.timestamp):
            if request.entries:
                for entry in request.entries:
                    if entry.key in self.lane_data:
                        if self.weighted:
                            total = request.internal_time + self.clock
                            self.lane_data[entry.key] = max((entry.value*(request.internal_time / total) + self.lane_data[entry.key]*(self.clock / total)), 0)
                        else:
                            self.lane_data[entry.key] = max((entry.value + self.lane_data[entry.key]) / 2, 0)
                    else:
                        self.lane_data[entry.key] = max(entry.value, 0)
        return traffic_pb2.DataReply(version=1, time=self.clock, error_code="200")
            


class TrafficLight:
    def __init__(self, 
                 port: int, 
                 config: dict, 
                 id: str, 
                 env: TrafficEnvironment, 
                 agent: PPO, 
                 model_lock: th.Lock, 
                 other_intersections: dict, 
                 intersection_tokens: dict, 
                 map_config: str = 'configs/roadnet2by3.json', 
                 use_model: bool = False):
        """
        Initialize a `TrafficLight` object.

        Args:
        - port (int): Port number on which to start the traffic light server.
        - config (dict): Configuration information for the traffic light.
        - id (str): Intersection ID for the traffic light.
        - env (TrafficEnvironment): `TrafficEnvironment` object representing the environment in which the traffic light exists.
        - agent (PPO): `PPO` object representing the reinforcement learning agent that controls the traffic light.
        - model_lock (th.Lock): `th.Lock` object used to synchronize access to the agent's neural network model.
        - other_intersections (dict): Dictionary of intersection IDs and their corresponding traffic light objects for all other intersections in the environment.
        - intersection_tokens (dict): Dictionary of intersection IDs and their corresponding API keys for authenticating data transmissions.
        - map_config (str): File path to the configuration file for the road network map.
        - use_model (bool): Whether to use the agent's neural network model to make decisions (default is `False`).

        Returns:
        - None
        """
        
        self.config = config
        self.port = port
        self.servicer = None
        self.use_model = use_model
        
        # global data
        self.phase_id = 0
        self.intersection_id = id
        self.env = env
        self.agent = agent
        self.model_lock = model_lock
        self.active = {inter_id:1 for inter_id in other_intersections}
        self.other_intersections = other_intersections
        self.broken = False
        self.intersection_tokens = intersection_tokens 
        
        # config dependent attrs
        with open(map_config) as f:
            self.layout = json.load(f)
        for intersection in self.layout['intersections']:
            if intersection_id == intersection['id']:
                self.intersection_config = intersection
        self.n_phases = len(self.intersection_config['trafficLight']['lightphases'])
        self.lightphases = self.intersection_config['trafficLight']['lightphases']
        self.roadlinks = self.intersection_config['roadLinks']
        
        # durartion measure
        self.phase_duration = 0
        self.allocated_phase_time = 5
        
        
    def serve(self):
        """
        Start the traffic light server and begin accepting incoming requests.

        Args:
        - None

        Returns:
        - None
        """

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self.servicer = TrafficServerServicer(port=self.port, 
                                              intersection_tokens=self.intersection_tokens)
        traffic_pb2_grpc.add_TrafficServerServicer_to_server(self.servicer, server)
        with open('certs/localhost.pem', 'rb') as f:
            private_key = f.read()

        with open('certs/localhost.crt', 'rb') as f:
            certificate_chain = f.read()

        server.add_secure_port(f"[::]:{self.port}", grpc.ssl_server_credentials([(private_key, certificate_chain)]))
        #server.add_secure_port(f"[::]:{self.port}", grpc.ssl_server_credentials([('private_key.pem', 'certificate_chain.pem')]))
        #grpc.ssl_server_credentials
        #server.add_secure_port()
        server.start()
        print("started", self.port)
        while not self.broken:
            pass
        print("stopped")


    # configure peer discovery and establish gRPC connections
    def find_peers(self):
        """
        Discover peer traffic lights in the environment.

        Args:
        - None

        Returns:
        - dict: Dictionary of peer traffic light configurations.
        """
        return self.config
    

    def send_data(self, data, stub):
        """
        Send traffic data to a peer traffic light.

        Args:
        - data (dict): Dictionary of traffic data to send.
        - stub (TrafficServerStub): gRPC stub for the peer traffic light server.

        Returns:
        - None
        """
        for key, val in data.items():
            self.servicer.lane_data[key] = val
        entries = [traffic_pb2.DataTransmission.KeyValue(key=k, value=v) for k, v in data.items()]
        request = traffic_pb2.DataTransmission(entries=entries, version=1, 
                                               intersection_id=self.intersection_id, 
                                               timestamp=str(datetime.datetime.now().timestamp()),
                                               api_key=self.intersection_tokens[self.intersection_id])
        response = stub.TransmitData(request, timeout=5.0)
        if response.error_code and response.error_code == "200":
            pass
        else:
            print("transmission failure")


    def connect_to_peers(self):
        """
        Establish gRPC connections with peer traffic lights.

        Args:
        - None

        Returns:
        - list: List of gRPC stubs for peer traffic light servers.
        """
        peers = self.find_peers()
        stub_list = []
        for peer in peers:
            # with open('private_key.pem', 'rb') as f:
            #     private_key = f.read()

            with open('certs/localhost.crt', 'rb') as f:
                certificate_chain = f.read()
                
            credentials = grpc.ssl_channel_credentials(
                root_certificates=certificate_chain,
            )
            channel = grpc.secure_channel('{}:{}'.format(peer[0], peer[1]), credentials)
            stub = traffic_pb2_grpc.TrafficServerStub(channel)
            stub_list.append(stub)
        return stub_list
    
    def get_lanes_for_intersection(self):
        """
        Get the list of all lanes that are either incoming or outgoing for the current intersection.

        Returns:
        --------
        lanes : List[str]
            List of lane IDs.
        """
        incoming_roads = self.env.map.get_intersection_incoming_roads(self.intersection_id)
        outgoing_roads = self.env.map.get_intersection_outgoing_roads(self.intersection_id)
        incoming_lanes = []
        outgoing_lanes = []
        for r in incoming_roads:
            incoming_lanes.extend(self.env.map.get_road_lanes(r))
        for r in outgoing_roads:
            outgoing_lanes.extend(self.env.map.get_road_lanes(r))
        lanes = list(set(incoming_lanes + outgoing_lanes))
        
        return lanes
    
    def send_heartbeat(self, stub):
        """
        Send a heartbeat message to the given stub.

        Parameters:
        -----------
        stub : object
            A gRPC stub object that is used to communicate with a remote intersection.

        Raises:
        -------
        AssertionError:
            If the response from the remote intersection contains an error code.
        """
        message = traffic_pb2.RefreshRequest(
                            version=1, time=self.servicer.clock, phase_id=self.phase_id, 
                            intersection_id=self.intersection_id, api_key=self.intersection_tokens[self.intersection_id])
        response = stub.HeartBeat(message, timeout=5.0)
        if response.error_code:
            assert response.error_code == "200"

    def update_loop(self):
        """
        Continuously send data updates to other intersections.

        This function runs in an infinite loop and sends data updates to all other active intersections.
        It retrieves the lane waiting vehicle count data from the environment, identifies the lanes that
        are either incoming or outgoing for the current intersection, and sends this data to all active
        intersections using gRPC.
        """
        stubs = self.connect_to_peers()
        while True:
            time.sleep(SERVER_REFRESH_TIME)
            # get lane data
            lane_data = self.env.eng.get_lane_waiting_vehicle_count()
            lanes = self.get_lanes_for_intersection()
            
            data_transmission = {l:lane_data[l] for l in lanes}
            N = len(stubs)
            for i in range(N):
                if self.active[self.other_intersections[i]]:
                    stub = stubs[i]
                    for _ in range(RETRY_ATTEMPTS):
                        try:
                            self.send_heartbeat(stub)
                            self.send_data(data_transmission, stub)
                        except Exception as e:
                                print("Caught an error")
                                failed_intersection_id = self.other_intersections[i]
                                self.active[failed_intersection_id] = 0
    
    def agent_model_control_logic(self):
        """Control logic for an agent that uses a machine learning model to make decisions.

        Uses a trained machine learning model to predict the next traffic light phase, based on 
        historical data from the intersection. If the allocated phase time has not elapsed, then
        the function updates the phase duration time. Otherwise, the function sets the new phase 
        ID and updates the allocated phase time.

        Args:
            None

        Returns:
            None
        """
        if self.allocated_phase_time <= self.phase_duration:
            data = self.env.history[self.intersection_id]
            # returns a tuple, the 0th element is the prediction
            phase = self.agent.predict(data)[0] 

            # resulting phase logic to control duration
            self.phase_id = phase * 2
            self.phase_duration = 0
            self.allocated_phase_time = 5
        else:
            self.phase_duration += 1
            
    def get_lightphase_info(self):
        """
        Collects lightphase information for the current intersection.

        Collects data about all of the available lightphases for the intersection. This data includes
        the names of the roads and lanes affected by each lightphase, as well as the current waiting 
        vehicle count on each lane.

        Args:
            None

        Returns:
            lightphase_w_cars_start (dict): A dictionary mapping lightphase IDs to a list of waiting vehicle 
                counts on the lanes that will be affected by the start of the lightphase.
            lightphase_w_cars_end (dict): A dictionary mapping lightphase IDs to a list of waiting vehicle 
                counts on the lanes that will be affected by the end of the lightphase.
        """
        # get lightphase data
        lightphase_w_cars_start = {}
        lightphase_w_cars_end = {}
        for phase in range(self.n_phases):
            affected_links = self.lightphases[phase]['availableRoadLinks']
            start_lane_names, end_lane_names = [], []
            for link in affected_links:
                start_road = self.roadlinks[link]['startRoad']
                start_lanes = list(set([lane['startLaneIndex'] for lane in self.roadlinks[link]['laneLinks']]))
                start_lane_names += [f"{start_road}_{lane}" for lane in start_lanes]
                
                end_road = self.roadlinks[link]['endRoad']
                end_lanes = list(set([lane['endLaneIndex'] for lane in self.roadlinks[link]['laneLinks']]))
                end_lane_names += [f"{end_road}_{lane}" for lane in end_lanes]
                #print(lane_names)
            lightphase_w_cars_start[phase] = [self.servicer.lane_data[name] for name in start_lane_names]
            lightphase_w_cars_end[phase] = [self.servicer.lane_data[name] for name in end_lane_names]
            
        return lightphase_w_cars_start, lightphase_w_cars_end
                
    def simple_control_logic(self):
        """
        Control logic for a traffic light that uses a simple algorithm to make decisions.

        Uses a simple algorithm to determine the next traffic light phase. If the allocated phase time 
        has not elapsed, then the function updates the phase duration time. Otherwise, the function 
        sets the new phase ID and updates the allocated phase time, based on the total number of 
        cars entering and exiting the intersection.

        Args:
            None

        Returns:
            None
        """
        if self.allocated_phase_time <= self.phase_duration:
            # set new phase id
            self.phase_id = (self.phase_id + 1) % self.n_phases
            self.phase_duration = 0
            
            if USE_FIXED_TIME:
                self.allocated_phase_time = 30 if self.allocated_phase_time == 5 else 5
            else:
                entering_cars, exiting_cars = self.get_lightphase_info()
                total_entering_cars = sum(entering_cars[self.phase_id])
                total_exiting_cars = sum(exiting_cars[self.phase_id])
                self.allocated_phase_time = min((total_entering_cars + STANDARD_PHASE_TIME - total_exiting_cars) / 
                                                CAR_DISCOUNT_FACTOR + MINIMUM_PHASE_TIME, MAXIMUM_PHASE_TIME)
        else:
            self.phase_duration += 1
            
    def run(self):
        print("Starting listening interface", self.port)
        # Start the thread for the listening interface
        init_thread = th.Thread(target=self.serve)
        init_thread.start()

        # Add delay to initialize the server-side logic on all processes
        time.sleep(2*INITIALIZE_WAIT_TIME)
        
        print("starting intercomms - checking internal state")
        print("Current Clock State: ", self.servicer.clock)
        
        update_thread = th.Thread(target=self.update_loop)
        update_thread.start()
        
        time.sleep(INITIALIZE_WAIT_TIME)
        
        while True:
            time.sleep(SERVER_REFRESH_TIME)
            self.servicer.clock += 1
            
            if self.use_model and not self.broken:
                self.agent_model_control_logic()
                
            elif self.broken:
                self.phase_id = self.servicer.clock % self.n_phases
            
            else:
                self.simple_control_logic()
            
            if self.servicer.clock == ITERATIONS:
                break
        
        init_thread.join()
        update_thread.join()
                
def generate_token():
        """
        Generates a token for authenticating user requests to a chat server.

        Returns:
            str: A token that can be used to authenticate user requests.
        """

        token = os.urandom(TOKEN_LENGTH)
        return binascii.hexlify(token).decode()              
      
if __name__ == '__main__':
    
    
    hosts = [('localhost', 50051), ('localhost', 50052), ('localhost', 50053), 
             ('localhost', 50054), ('localhost', 50055), ('localhost', 50056)]
    
    # broken_config = {"intersection_1_1":150, "intersection_3_2": 150} # example
    broken_config = {}
    
    config_path = "configs/config.json"
    env = TrafficEnvironment(30, False, config_path, thread_num=4)
    eng = env.eng
    # eng = cityflow.Engine(config_path, thread_num=1)

    # load model
    model_path = "ppo_dummy.zip"
    env.time_per_episode = 1000.0
    model = PPO("MlpPolicy", env, verbose=1)
    model.load(model_path)
    
    # generate config assignment
    id_configs = [(i, j) for i in range(1,4) for j in range(1,3)]
    N = len(hosts)
    lights = []
    model_lock = th.Lock()    
    
    # setup each light config
    intersection_ids = [f"intersection_{val1}_{val2}" for (val1,val2) in id_configs]
    
    # authorization tokens
    intersection_tokens = {intersection_id: generate_token() for intersection_id in intersection_ids}
    
    for i in range(N):
        val1, val2 = id_configs[i]
        intersection_id = intersection_ids[i]
        config = hosts[:i] + hosts[i+1:]
        other_intersections = intersection_ids[:i] + intersection_ids[i+1:]
        light = TrafficLight(port=hosts[i][1], 
                              config=config, 
                              id=intersection_id,
                              env=env,
                              agent=model,
                              model_lock=model_lock,
                              other_intersections=other_intersections,
                              intersection_tokens=intersection_tokens,
                              use_model=USE_MODEL
                              )
        lights.append(light)
        
    # threads for traffic lights - start and store 'em
    light_threads = []
    for i in range(N):  
        light_thread = th.Thread(target=lights[i].run)
        light_threads.append(light_thread)
        light_thread.start()

    for i in tqdm(range(ITERATIONS)):
        eng.next_step()
        env.update_history()
        for j in range(N):
            intersection_id = lights[j].intersection_id
            if intersection_id in broken_config and broken_config[intersection_id] == i:
                lights[j].broken = True
                print("broken")
            phase_id = lights[j].phase_id
            eng.set_tl_phase(intersection_id, phase_id)
        time.sleep(SERVER_REFRESH_TIME)
        
    print("DONE")
    
    print(f"Average Travel Time: {eng.get_average_travel_time()}")
    
    # join remaining threads
    for i in range(N):  
        light_threads[i].join()    
