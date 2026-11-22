from dataclasses import dataclass, field
import math
import random
import numpy as np

@dataclass
class Agent:
    agent_id: int = 0
    agent_type: np.ndarray = field(default_factory=lambda: np.array([]))
    availability: float = 0; x: float = 0; y: float = 0; z: float = 0
    nom_velocity: float = 1; fuel: float = 0

    @property
    def task_id(self):
        return self.agent_id

@dataclass
class Task:
    task_id: int = 0
    task_type: np.ndarray = field(default_factory=lambda: np.array([]))
    task_value: float = 60; start_time: float = 0; end_time: float = 0
    duration: float = 60; discount: float = 0.99
    x: float = 0; y: float = 0; z: float = 0

    @property
    def agent_id(self):
        return -1

@dataclass
class WorldInfo:
    limit_x: list = field(default_factory=lambda: [0, 100])
    limit_y: list = field(default_factory=lambda: [0, 100])
    limit_z: list = field(default_factory=lambda: [0, 0])
    distance_max: float = 0
    def __post_init__(self):
        self.distance_max = math.sqrt((self.limit_x[1] - self.limit_x[0])**2 + (self.limit_y[1] - self.limit_y[0])**2 + (self.limit_z[1] - self.limit_z[0])**2)

def create_agents_and_tasks(Q, Y, world_info, agent_coordinates, task_coordinates):
    num_agents = Q.shape[1]
    num_tasks = Y.shape[1]
    agent_list = []
    task_list = []
    for idx_agent in range(num_agents):
        agent = Agent(agent_id=idx_agent, agent_type=Q[:, idx_agent])
        agent.nom_velocity = 2.0
        if agent_coordinates and len(agent_coordinates) > idx_agent:
            agent.x, agent.y = agent_coordinates[idx_agent]
        else:
            agent.x, agent.y = 0, 0
        agent_list.append(agent)
    for idx_task in range(num_tasks):
        task = Task(task_id=idx_task, task_type=Y[:, idx_task])
        if task_coordinates and len(task_coordinates) > idx_task:
            task.x, task.y = task_coordinates[idx_task]
        else:
            task.x = random.uniform(world_info.limit_x[0], world_info.limit_x[1])
            task.y = random.uniform(world_info.limit_y[0], world_info.limit_y[1])
        task.start_time = 0
        task.end_time = 10000
        task_list.append(task)
    return agent_list, task_list
