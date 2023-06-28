# Distributed Traffic Control
## Authors: Aakash Mishra and Rajat Mittal (@aakamishra, @Green-Fantasy)
## Distributed Intelligent Traffic Control

We present a novel approach for mitigating traffic congestion using a distributed reinforcement learning model that is fault-tolerant and secure. Traditional methods of traffic signal timing optimization often rely on fixed timing schedules, which fail to adapt to real-time traffic conditions and can lead to increased congestion. In contrast, our proposed system leverages the power of distributed reinforcement learning to dynamically optimize traffic signal timings based on the current traffic patterns. The distributed nature of our model ensures fault tolerance and resilience in the face of individual component failures. By distributing the learning and decision-making process across multiple agents, the system can continue to operate effectively even if some agents become unresponsive or encounter failures. This fault-tolerant design minimizes the impact of system disruptions, improving the overall reliability of traffic signal optimization. Furthermore, security is a critical aspect of our approach, considering the potential implications of tampering or malicious attacks on traffic systems. Towards this we present a system with several safeguards in place to prevent malicious attacks. To evaluate the effectiveness of our approach, simulations were conducted using a 3x2 intersection square grid traffic network. The results demonstrate significant improvements in traffic congestion compared to the traditional fixed timing method, achieving a 22\% reduction in congestion levels.

## Setup

Setup for codebase. In order to run the setup script, create a new conda or PyPI environment 

```
conda create -n traffic-control python=3.9
```

We can then activate this environment as follows. 

```
conda activate traffic-control
```

We can then install the dependencies using the `requirements.txt` file.

```
pip install -r requirements.txt
```

## Running and Testing

In order to run the simulator please run:

```
python traffic_server.py
```

In order to run unit tests please run the following:

```
python unit_tests.py
```

If sucessful, there should be a print statement at the end that enumerates whether the tests passed or not.

## Notebook

[Engineering Notebook](engineering_notebook.md)

## Visualization

For intructions on how to visualize the traffic data, please see the CityFlow repository [here](https://cityflow-project.github.io/).


