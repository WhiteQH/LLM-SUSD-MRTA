import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def path_planning(p_r, P_t, bundle):
    num_tasks = P_t.shape[0]
    if num_tasks == 0: return [], 0
    Dist = pairwise_distances(P_t)
    Dist_robot = pairwise_distances(p_r, P_t)
    local_idx = np.argmin(Dist_robot[0, :])
    goal_ordering = [bundle[local_idx]]
    Dist[:, local_idx] = np.inf
    total_dist = Dist_robot[0, local_idx]
    for _ in range(num_tasks - 1):
        next_local_idx = np.argmin(Dist[local_idx, :])
        total_dist += Dist[local_idx, next_local_idx]
        goal_ordering.append(bundle[next_local_idx])
        local_idx = next_local_idx
        Dist[:, local_idx] = np.inf
    return goal_ordering, total_dist

def bundle_exec_time(agent, bundle, task_list, specialization_mat):
    p_r = np.array([[agent.x, agent.y, agent.z]])
    if not bundle: return [], 0
    P_t = np.array([[task_list[tid].x, task_list[tid].y, task_list[tid].z] for tid in bundle])
    goal_ordering, total_dist = path_planning(p_r, P_t, bundle)
    travel_time = total_dist / agent.nom_velocity if agent.nom_velocity > 0 else float('inf')
    task_time = sum(task_list[tid].duration / specialization_mat[agent.agent_id, tid] for tid in bundle)
    return goal_ordering, travel_time + task_time
