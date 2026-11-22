import numpy as np

def merge_planning_and_makespan(agent_position, Q, task_position, Y, robot_task, base_time=60.0):
    n_agents = Q.shape[1]
    if len(robot_task) != n_agents:
        print(f"Warning: Mismatch in robot count. Q matrix implies {n_agents}, but allocation has {len(robot_task)}. Adjusting...")
        while len(robot_task) < n_agents:
            robot_task.append([])
        robot_task = robot_task[:n_agents]

    n_tasks = task_position.shape[0]
    robot_types = [2 if Q[0, i] < Q[1, i] else 1 for i in range(n_agents)]
    task_types = [2 if Y[1, j] == 1 else 1 for j in range(n_tasks)]

    def greedy_route(task_indices, start_pos, all_task_pos):
        current_pos = np.array(start_pos)
        route, remaining = [], task_indices.copy()
        while remaining:
            distances = [np.linalg.norm(all_task_pos[idx] - current_pos) for idx in remaining]
            nearest = remaining[np.argmin(distances)]
            route.append(nearest)
            current_pos = all_task_pos[nearest]
            remaining.remove(nearest)
        return route

    sorted_robot_task = [greedy_route(tasks, agent_position[i], task_position) if tasks else [] for i, tasks in enumerate(robot_task)]

    agent_times = []
    for i in range(n_agents):
        current_pos = np.array(agent_position[i])
        total_time = 0.0
        r_type, speed = robot_types[i], 2.0
        for task_idx in sorted_robot_task[i]:
            task_pos = np.array(task_position[task_idx])
            travel_time = np.linalg.norm(task_pos - current_pos) / speed
            exec_time = base_time / 2 if r_type == task_types[task_idx] else base_time
            total_time += travel_time + exec_time
            current_pos = task_pos
        agent_times.append(total_time)

    return sorted_robot_task, (max(agent_times) if agent_times else 0.0), agent_times
