import numpy as np
import ast
import os
import re
import openai
import sys
import time
from .path_planning import bundle_exec_time
from .greedy_validation import merge_planning_and_makespan
from .data_management import Agent, Task

class SUSD:
    def __init__(self, Q, Y):
        self.agent_types = Q; self.task_types = Y
        self.specialization_mat = np.matmul(Q.transpose(), Y)
        self.num_agents = Q.shape[1]; self.num_tasks = Y.shape[1]
        self.agent_list = None; self.task_list = None

    def settings(self, agent_list, task_list):
        self.agent_list = agent_list; self.task_list = task_list

    def initiliza_Theta(self, ini_bundle_list, ini_bias):
        ini_Theta = np.random.uniform(0, 0.1, (self.num_agents, self.num_tasks))
        for i in range(self.num_agents):
            if ini_bundle_list[i]: ini_Theta[i, ini_bundle_list[i]] += ini_bias
        return ini_Theta

    def solve(self, agent_list, task_list, ini_bundle_list, T=100, ini_bias=10, cancellation_token=None):
        start_time = time.perf_counter()
        self.settings(agent_list, task_list)
        ini_Theta = self.initiliza_Theta(ini_bundle_list, ini_bias)
        k1, k2, d_des, self.eps, num_samples = 2.0, 1.0, 0.2, 0.2, 8
        _, c0 = self.evaluate_bundles(ini_bundle_list)
        best_bundle_list = ini_bundle_list
        print(f"start {best_bundle_list} {c0}", flush=True)

        num_param = self.num_agents * self.num_tasks
        if num_param == 0: return best_bundle_list, 0, 0
        num_vagents = 2 * num_param
        Theta_ = np.zeros((num_vagents, num_param))
        for v in range(num_vagents):
            Theta_[v, :] = (ini_Theta + np.random.normal(0.0, self.eps, ini_Theta.shape)).flatten()
        Theta_[0, :] = ini_Theta.flatten()
        v_old = None
        for t in range(T):
            if cancellation_token and cancellation_token(): return "Cancelled", -1, 0
            C = np.zeros(num_vagents)
            for v in range(num_vagents):
                bundle_list_v = self.Theta_to_bundles(Theta_[v, :].reshape(ini_Theta.shape))
                _, c_v = self.evaluate_bundles(bundle_list_v)
                if c_v < c0:
                    c0 = c_v; best_bundle_list = bundle_list_v
                    print(t, c0, flush=True)
                C[v] = c_v
            w, v_eig = np.linalg.eigh(np.cov(Theta_.T))
            v_1 = v_eig[:, 0]
            c_min, c_arg = np.min(C), np.argmin(C)
            Theta_mean = np.mean(Theta_, axis=0)
            z = 1 - np.exp(c_min - C)
            if t == 0: v_old = v_1
            if v_old is not None and np.dot(v_1, v_old) < 0: v_1 = -v_1
            v_old = v_1
            d = Theta_ - Theta_mean
            d_norm = np.linalg.norm(d, axis=1, keepdims=True); d_norm[d_norm == 0] = 1
            U = d * (d_des - d_norm) / (d_norm**2)
            Theta_ = Theta_ + k1 * (z.reshape(-1, 1) * v_1.reshape(1, -1)) + k2 * U
            for _ in range(num_samples):
                bundle = self.Theta_to_bundles(Theta_[c_arg, :].reshape(ini_Theta.shape))
                _, c = self.evaluate_bundles(bundle)
                if c < c0:
                    c0 = c; best_bundle_list = bundle
                    print(f"{t} {c0} {best_bundle_list}", flush=True)
            Theta_ += np.random.normal(0.0, 1, Theta_.shape)
        final_paths, final_makespan = self.evaluate_bundles(best_bundle_list)
        end_time = time.perf_counter()
        return final_paths, final_makespan, end_time - start_time

    def Theta_to_bundles(self, Theta):
        bundle_list = [[] for _ in range(self.num_agents)]
        for j in range(self.num_tasks):
            theta = Theta[:, j]; ex_theta = np.exp(theta)
            ptheta = ex_theta / np.sum(ex_theta)
            bundle_list[np.random.choice(self.num_agents, p=ptheta)].append(j)
        return bundle_list

    def evaluate_bundles(self, bundle_list):
        exec_time_vec, path_list = np.zeros(self.num_agents), [[] for _ in range(self.num_agents)]
        for i, agent in enumerate(self.agent_list):
            paths, exec_time = bundle_exec_time(agent, bundle_list[i], self.task_list, self.specialization_mat)
            path_list[i] = paths; exec_time_vec[i] = exec_time
        return path_list, np.max(exec_time_vec) if exec_time_vec.size > 0 else 0

def configure_openai(api_key, api_endpoint, deployment_name):
    openai.api_key=api_key; openai.api_base=api_endpoint; openai.api_type="azure"; openai.api_version="2023-05-15"
    return deployment_name

def get_allocation_from_llm(prompt, deployment_name, validate_func, max_retries=10):
    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            response = openai.ChatCompletion.create(
                engine=deployment_name,
                messages=[{"role": "system", "content": "You are an expert in multi-robot task allocation."},
                          {"role": "user", "content": prompt}],
                max_tokens=1500, temperature=0.7)
            end_time = time.perf_counter()
            llm_time = end_time - start_time
            llm_output = response['choices'][0]['message']['content'].strip()

            start_index = llm_output.find('[')
            end_index = llm_output.rfind(']')

            if start_index != -1 and end_index > start_index:
                list_str = llm_output[start_index:end_index+1]
                try:
                    allocation = ast.literal_eval(list_str)
                    is_valid, msg = validate_func(allocation)
                    if is_valid:
                        return allocation, llm_time
                    else:
                        print(f"Invalid LLM allocation: {msg} (Attempt {attempt + 1}/{max_retries})", flush=True)
                except (ValueError, SyntaxError):
                    print(f"LLM response could not be parsed. (Attempt {attempt + 1}/{max_retries})", flush=True)
            else:
                print(f"Could not find list in LLM response. (Attempt {attempt + 1}/{max_retries})", flush=True)

        except Exception as e:
            print(f"Error during LLM API call (Attempt {attempt+1}/{max_retries}): {e}", flush=True)

    print(f"Exceeded maximum retries for LLM allocation.", flush=True)
    return None, 0

def validate_allocation(allocation, num_robots, total_tasks):
    if allocation is None:
        return False, "Allocation is None"
    if not isinstance(allocation, (list, tuple)):
        return False, "Allocation should be a list or tuple of robot task lists"
    if len(allocation) != num_robots:
        return False, f"Allocation should have exactly {num_robots} robots, found {len(allocation)}"

    all_tasks = []
    for i, robot_tasks in enumerate(allocation):
        if not isinstance(robot_tasks, (list, tuple)):
            return False, f"Tasks for Robot {i+1} should be a list or tuple"
        if len(robot_tasks) == 0:
            return False, f"Robot {i+1} has been assigned no tasks"
        all_tasks.extend(robot_tasks)

    unique_tasks = set(all_tasks)
    if len(all_tasks) != total_tasks:
        return False, f"Total number of assigned tasks is {len(all_tasks)}, but expected {total_tasks}"

    if len(unique_tasks) != total_tasks:
        return False, f"Found duplicate or missing tasks (unique tasks: {len(unique_tasks)}, total tasks: {total_tasks})"

    if not all(0 <= task < total_tasks for task in all_tasks):
        return False, "Found tasks outside valid range"

    return True, "Valid allocation"

def get_prompt_content(config, task_list, agent_list, feedback_data=None):
    num_robots = len(agent_list)
    num_tasks = len(task_list)
    q_matrix = np.array(config['robot_params']['q_matrix'])

    type1_robots = [str(i) for i in range(num_robots) if np.array_equal(q_matrix[:, i], [1, 2])]
    type2_robots = [str(i) for i in range(num_robots) if np.array_equal(q_matrix[:, i], [2, 1])]

    robot_info_lines = []
    if type1_robots:
        robot_info_lines.append(f"- Robot(s) {', '.join(type1_robots)} (Lightweight AUVs) are primarily suited for survey tasks.")
    if type2_robots:
        robot_info_lines.append(f"- Robot(s) {', '.join(type2_robots)} (Hovering AUVs) are primarily suited for inspection tasks.")
    robot_info_lines.append("All robots start at initial coordinates [0,0].")
    robot_info = "\n".join(robot_info_lines)

    inspection_tasks = [str(i) for i in range(num_tasks) if i % 2 != 0]
    survey_tasks = [str(i) for i in range(num_tasks) if i % 2 == 0]
    task_info_lines = []
    if inspection_tasks:
        task_info_lines.append(f"- Inspection tasks: {', '.join(inspection_tasks)}.")
    if survey_tasks:
        task_info_lines.append(f"- Survey tasks: {', '.join(survey_tasks)}.")
    task_info = "\n".join(task_info_lines)

    task_pos_str = '\n'.join([f"Task {t.task_id}: {np.round(np.array([t.x, t.y]), 4).tolist()}" for t in task_list])
    agent_pos_str = '\n'.join([f"Agent {a.agent_id}: {np.round(np.array([a.x, a.y]), 4).tolist()}" for a in agent_list])

    format_example_lines = ["    [tasks_for_robot_{}],".format(i) for i in range(num_robots)]
    format_example = "[\n" + "\n".join(format_example_lines) + "\n]"

    params = {
        'TOTAL_TASKS': num_tasks, 
        'TOTAL_TASKS_MINUS_1': num_tasks - 1 if num_tasks > 0 else 0,
        'NUM_ROBOTS': num_robots,
        'robot_info': robot_info, 
        'task_info': task_info,
        'task_pos_str': task_pos_str, 
        'agent_pos_str': agent_pos_str,
        'FORMAT_EXAMPLE': format_example
    }

    if feedback_data:
        template = config['prompts']['feedback_prompt']
        alloc_strs = [f"iteration {i+1} LLM generates an allocation as {d['Na']}, a better optimisation based on this allocation is {d['Nb']}, and makespan is {d['makespan']:.2f}." for i, d in enumerate(feedback_data)]

        print("\n--- Debug Info (get_feedback_prompt) ---", flush=True)
        print("allocations_str:\n" + "\n".join(alloc_strs), flush=True)

        params['allocations_str'] = '\n'.join(alloc_strs)
        params['latest_Nb'] = feedback_data[-1]['Nb']
    else:
        template = config['prompts']['initial_prompt']

    return template.format(**params)


def run_llm_susd_flow(config, agent_list, task_list, run_group_id, cancellation_token=None):

    print("\nTask Coordinates:", flush=True)
    for t in task_list: print(f"  Task {t.task_id}: {np.round(np.array([t.x, t.y]), 4).tolist()}", flush=True)
    print("\nAgent Coordinates:", flush=True)
    for a in agent_list: print(f"  Agent {a.agent_id}: {[a.x, a.y]}", flush=True)

    api_key = os.getenv("AZURE_OPENAI_KEY", config['api_keys'].get('azure_openai_key', "YOUR_AZURE_OPENAI_KEY_HERE"))
    api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", config['api_keys'].get('azure_openai_endpoint', "https://YOUR-RESOURCE.openai.azure.com/"))
    deployment_name = configure_openai(api_key, api_endpoint, config['api_keys']['azure_deployment_name'])
    Q = np.array(config['robot_params']['q_matrix'])
    num_tasks = len(task_list)
    Y = np.zeros((2, num_tasks)); Y[0, :] = [1 if i % 2 == 0 else 0 for i in range(num_tasks)]; Y[1, :] = 1 - Y[0, :]
    agent_pos = np.array(config['robot_params']['start_coords'])
    task_pos = np.array([[t.x, t.y] for t in task_list])

    susd_solver = SUSD(Q, Y)
    results = []

    validator = lambda alloc: validate_allocation(alloc, len(agent_list), len(task_list))

    # Iteration 1
    prompt = get_prompt_content(config, task_list, agent_list)
    print("\n" + "="*20 + " PROMPT FOR LLM (ITERATION 1) " + "="*20, flush=True)
    print(prompt, flush=True)
    print("="*60 + "\n", flush=True)
    Na, llm_time = get_allocation_from_llm(prompt, deployment_name, validator, config['solvers']['llm_max_retries'])
    if Na is None: return "Failed to get initial LLM allocation."
    if cancellation_token and cancellation_token(): return "Cancelled"

    A2, makespan_A2, times_A2 = merge_planning_and_makespan(agent_pos, Q, task_pos, Y, Na)

    print(f"\n[Group {run_group_id}- LLM] Iteration 1:", flush=True)
    print("Allocation (LLM):", flush=True)
    print(Na, flush=True)
    print(f"Computed Makespan (A2): {makespan_A2:.2f} seconds", flush=True)
    print("Individual Agent Times:", [f"{t:.2f}" for t in times_A2], flush=True)

    print("\nAllocation (LLM+PP) (A2):", flush=True)
    print(A2, flush=True)
    print(f"Computed Makespan (A2): {makespan_A2:.2f} seconds", flush=True)
    print("Individual Agent Times:", [f"{t:.2f}" for t in times_A2], flush=True)

    A3, makespan_A3, susd_time = susd_solver.solve(agent_list, task_list, A2, T=config['solvers']['susd_iterations'], cancellation_token=cancellation_token)
    if A3 == "Cancelled": return "Cancelled"

    print("\nAllocation (LLM+PP+SUSD) (A3):", flush=True)
    print(A3, flush=True)
    A4, makespan_A4, times_A4 = merge_planning_and_makespan(agent_pos, Q, task_pos, Y, A3)
    print(f"Computed Makespan (A4):{makespan_A4:.2f} seconds", flush=True)
    print("Individual Agent Times:",[f"{t:.2f}"for t in times_A4], flush=True)

    print("\nAllocation (LLM+PP+SUSD+PathPlanning) (A4):", flush=True)
    print(A4, flush=True)
    print(f"Computed Makespan (A4): {makespan_A4:.2f} seconds", flush=True)
    print("Individual Agent Times:", [f"{t:.2f}" for t in times_A4], flush=True)

    results.append({
        'Group': run_group_id, 'Iteration': 1, 'A1': Na, 'A2': A2, 
        'Computed_Makespan of A2': makespan_A2, 'A3': A3, 'A4': A4, 
        'Computed_Makespan of A4': makespan_A4, 
        'LLM_Time': llm_time, 'SUSD_Time': susd_time,
        'Task_Coordinates': task_pos.tolist(), 
        'Agent_Coordinates': agent_pos.tolist()
    })

    optimized_allocations = [{"Na": Na, "Nb": A3, "makespan": makespan_A4}]

    # Subsequent iterations
    for i in range(2, config['solvers']['llm_iterations'] + 1):
        if cancellation_token and cancellation_token(): return "Cancelled"

        prompt = get_prompt_content(config, task_list, agent_list, feedback_data=optimized_allocations)
        print("\n" + "="*20 + f" PROMPT FOR LLM (ITERATION {i}) " + "="*20, flush=True)
        print(prompt, flush=True)
        print("="*60 + "\n", flush=True)
        Na_new, llm_time_new = get_allocation_from_llm(prompt, deployment_name, validator, config['solvers']['llm_max_retries'])
        if Na_new is None: print(f"Failed to get valid LLM allocation for iteration {i}. Skipping.", flush=True); continue

        A2_new, makespan_A2_new, times_A2_new = merge_planning_and_makespan(agent_pos, Q, task_pos, Y, Na_new)

        print(f"\n[Group {run_group_id}- LLM] Iteration {i}:", flush=True)
        print("Allocation (LLM):", flush=True)
        print(Na_new, flush=True)
        print("\nAllocation (LLM+PP):", flush=True)
        print(A2_new, flush=True)
        print(f"Computed Makespan (A2): {makespan_A2_new:.2f} seconds", flush=True)
        print("Individual Agent Times:", [f"{t:.2f}" for t in times_A2_new], flush=True)

        A3_new, makespan_A3_new, susd_time_new = susd_solver.solve(agent_list, task_list, A2_new, T=config['solvers']['susd_iterations'], cancellation_token=cancellation_token)
        if A3_new == "Cancelled": return "Cancelled"

        print("\nAllocation (LLM+PP+SUSD) (A3):", flush=True)
        print(A3_new, flush=True)
        A4_new, makespan_A4_new, times_A4_new = merge_planning_and_makespan(agent_pos, Q, task_pos, Y, A3_new)
        print(f"Computed Makespan (A3): {makespan_A4_new:.2f} seconds", flush=True)
        print("Individual Agent Times:", [f"{t:.2f}" for t in times_A4_new], flush=True)

        print("\nAllocation (LLM+PP+SUSD+PathPlanning) (A4):", flush=True)
        print(A4_new, flush=True)
        print(f"Computed Makespan (A4): {makespan_A4_new:.2f} seconds", flush=True)
        print("Individual Agent Times:", [f"{t:.2f}" for t in times_A4_new], flush=True)

        results.append({
            'Group': run_group_id, 'Iteration': i, 'A1': Na_new, 'A2': A2_new, 
            'Computed_Makespan of A2': makespan_A2_new, 'A3': A3_new, 'A4': A4_new, 
            'Computed_Makespan of A4': makespan_A4_new, 
            'LLM_Time': llm_time_new, 'SUSD_Time': susd_time_new,
            'Task_Coordinates': task_pos.tolist(), 
            'Agent_Coordinates': agent_pos.tolist()
        })

        optimized_allocations.append({"Na": Na_new, "Nb": A3_new, "makespan": makespan_A4_new})
        if len(optimized_allocations) > config['solvers']['llm_max_context']:
            optimized_allocations.pop(0)

    return results
