from concurrent import futures
import grpc
import time
import traffic_pb2
import traffic_pb2_grpc
import threading as th

ft = '{}:{}'.format('localhost', '50051')
channel = grpc.insecure_channel(ft)
stub = traffic_pb2_grpc.TrafficServerStub(channel)
response = stub.HeartBeat(traffic_pb2.RefreshRequest(version=1, time=0))
print(response.message)