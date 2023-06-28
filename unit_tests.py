from traffic_server import *
from unittest.mock import MagicMock


def test_heartbeat_valid():
    servicer = TrafficServerServicer(port=8080, intersection_tokens={"intersection_id": "api_key"})
    request = traffic_pb2.RefreshRequest(intersection_id="intersection_id", api_key="api_key")
    response = servicer.HeartBeat(request, None)
    assert response.error_code == "200"
    
def test_heartbeat_invalid():
    servicer = TrafficServerServicer(port=8080, intersection_tokens={"intersection_id": "api_key"})
    request = traffic_pb2.RefreshRequest(intersection_id="invalid_id", api_key="invalid_key")
    response = servicer.HeartBeat(request, None)
    assert response.error_code == "432"
    
def test_transmitdata_new_timestamp():
    servicer = TrafficServerServicer(port=8080, intersection_tokens={"intersection_id": "api_key"})
    servicer.last_updated = 0.0
    request = traffic_pb2.DataTransmission(timestamp="1.0", entries=[traffic_pb2.DataTransmission.KeyValue(key="key1", value=2)], intersection_id="intersection_id", api_key="api_key")
    response = servicer.TransmitData(request, None)
    assert response.error_code == "200"
    assert servicer.last_updated == 1.0
    assert servicer.lane_data == {"key1": 2}

def test_transmitdata_existing_timestamp():
    servicer = TrafficServerServicer(port=8080, intersection_tokens={"intersection_id": "api_key"})
    servicer.last_updated = 1.0
    servicer.lane_data = {"key1": 1}
    request = traffic_pb2.DataTransmission(timestamp="1.0", entries=[traffic_pb2.DataTransmission.KeyValue(key="key1", value=2)], intersection_id="intersection_id", api_key="api_key")
    response = servicer.TransmitData(request, None)
    assert response.error_code == "200"
    assert servicer.last_updated == 1.0
    assert servicer.lane_data == {"key1": 1.5}
    
def test_heartbeat_valid_request():
        request = traffic_pb2.RefreshRequest(intersection_id='intersection_1_1', api_key='token1')
        context = MagicMock()
        intersection_tokens = {
            'intersection_1_1': 'token1',
            'intersection_1_2': 'token2',
            'intersection_2_1': 'token3',
            'intersection_2_2': 'token4'
        }
        servicer = TrafficServerServicer(50051, intersection_tokens)
        response = servicer.HeartBeat(request, context)

        assert response.version == 1
        assert response.time == 0
        assert response.error_code == "200"
        
def test_heartbeat_update_phase_status():
        intersection_tokens = {
                'intersection_1_1': 'token1',
                'intersection_1_2': 'token2',
                'intersection_2_1': 'token3',
                'intersection_2_2': 'token4'
            }
        servicer = TrafficServerServicer(50051, intersection_tokens)
        request = traffic_pb2.RefreshRequest(intersection_id='intersection_1_1', api_key='token1', phase_id=1)
        context = MagicMock()
        servicer.HeartBeat(request, context)

        assert servicer.phase_status['intersection_1_1'] == 1
        
def test_transmit_data_valid_request():
        intersection_tokens = {
                    'intersection_1_1': 'token1',
                    'intersection_1_2': 'token2',
                    'intersection_2_1': 'token3',
                    'intersection_2_2': 'token4'
                }
        servicer = TrafficServerServicer(50051, intersection_tokens)
        servicer.last_updated = 0.0
        request = traffic_pb2.DataTransmission(
            intersection_id='intersection_1_1',
            api_key='token1',
            timestamp='1.0',
            entries=[traffic_pb2.DataTransmission.KeyValue(key='lane1', value=10)]
        )
        context = MagicMock()
        response = servicer.TransmitData(request, context)

        assert response.version == 1
        assert response.time == 0
        assert response.error_code == "200"
        assert servicer.last_updated == 1.0
        assert servicer.lane_data['lane1'] == 10
        
def test_transmit_data_missing_fields():
        intersection_tokens = {
                        'intersection_1_1': 'token1',
                        'intersection_1_2': 'token2',
                        'intersection_2_1': 'token3',
                        'intersection_2_2': 'token4'
                    }
        servicer = TrafficServerServicer(50051, intersection_tokens)
        servicer.last_updated = 0.0
        request = traffic_pb2.DataTransmission(
            intersection_id='intersection_1_1',
            entries=[traffic_pb2.DataTransmission.KeyValue(key='lane1', value=10)]
        )
        context = MagicMock()
        response = servicer.TransmitData(request, context)
        assert response.version == 1
        assert response.time == 0
        assert response.error_code == "432"
        assert servicer.last_updated == 0.0
        assert 'lane1' not in servicer.lane_data
        
def test_transmit_negative_value_data():
        intersection_tokens = {
                    'intersection_1_1': 'token1',
                    'intersection_1_2': 'token2',
                    'intersection_2_1': 'token3',
                    'intersection_2_2': 'token4'
                }
        servicer = TrafficServerServicer(50051, intersection_tokens)
        servicer.last_updated = 0.0
        request = traffic_pb2.DataTransmission(
            intersection_id='intersection_1_1',
            api_key='token1',
            timestamp='1.0',
            entries=[traffic_pb2.DataTransmission.KeyValue(key='lane1', value=-10)]
        )
        context = MagicMock()
        response = servicer.TransmitData(request, context)

        assert response.version == 1
        assert response.time == 0
        assert response.error_code == "200"
        assert servicer.last_updated == 1.0
        assert servicer.lane_data['lane1'] == 0
        
def test_get_lightphase_info():
        # Mock the necessary objects and data
        intersection_tokens = {
                    'intersection_1_1': 'token1',
                    'intersection_1_2': 'token2',
                    'intersection_2_1': 'token3',
                    'intersection_2_2': 'token4'
                }
        servicer = TrafficServerServicer(50051, intersection_tokens)
        traffic_light = MagicMock()
        traffic_light.servicer = servicer
        traffic_light.n_phases = 2
        traffic_light.lightphases = [
            {'availableRoadLinks': [0, 1]},
            {'availableRoadLinks': [0]}
        ]
        traffic_light.roadlinks = {
            0: {'startRoad': 'road_1', 'endRoad': 'road_4', 'laneLinks': [{'startLaneIndex': 0, 'endLaneIndex': 6}, {'startLaneIndex': 1, 'endLaneIndex': 7}]},
            1: {'startRoad': 'road_2', 'endRoad': 'road_5', 'laneLinks': [{'startLaneIndex': 2, 'endLaneIndex': 8}]},
        }
        
        traffic_light.servicer.lane_data = {
            'road_1_0': [1, 2, 3],
            'road_1_1': [4, 5, 6],
            'road_2_2': [7, 8, 9],
            'road_3_3': [10, 11, 12],
            'road_3_4': [13, 14, 15],
            'road_3_5': [16, 17, 18],
            'road_4_6': [19, 20, 21],
            'road_4_7': [22, 23, 24],
            'road_5_8': [25, 26, 27],
            'road_6_9': [28, 29, 30],
            'road_6_10': [31, 32, 33]
        }

        # Expected output
        expected_start = {0: [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1: [[1, 2, 3], [4, 5, 6]]}
        expected_end = {0: [[19, 20, 21], [22, 23, 24], [25, 26, 27]], 1: [[19, 20, 21], [22, 23, 24]]}

        # Call the method to test
        lightphase_w_cars_start, lightphase_w_cars_end = TrafficLight.get_lightphase_info(traffic_light)

        # Assert the results
        assert lightphase_w_cars_start == expected_start
        assert lightphase_w_cars_end == expected_end
    
if __name__ == "__main__":
    print("Running Unit Tests:\ntest_heartbeat_valid()\n" +
          "test_heartbeat_invalid()\n" + 
          "test_transmitdata_existing_timestamp()\n" + 
          "test_transmitdata_new_timestamp()\n" + 
          "test_heartbeat_valid_request()\n" + 
          "test_heartbeat_update_phase_status()\n" + 
          "test_transmit_data_valid_request()\n" +
          "test_transmit_data_missing_fields()\n" +
          "test_transmit_negative_value_data()\n" +
          "test_get_lightphase_info()")
    test_heartbeat_valid()
    test_heartbeat_invalid()
    test_transmitdata_existing_timestamp()
    test_transmitdata_new_timestamp()
    test_heartbeat_valid_request()
    test_heartbeat_update_phase_status()
    test_transmit_data_valid_request()
    test_transmit_data_missing_fields()
    test_transmit_negative_value_data()
    test_get_lightphase_info()
    print("\033[92m [Traffic-Control Unit Test] 30 out 30 assert statements passed! \033[0m")
