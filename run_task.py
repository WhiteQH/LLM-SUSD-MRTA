import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import ast
import traceback
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.config_manager import load_config
from src.data_management import WorldInfo, create_agents_and_tasks
from src.llm_susd_module import run_llm_susd_flow

def run_single_task(config_path, params_override):
    total_start_time = time.perf_counter()
    try:
        print(f"--- [子进程] 开始处理配置: {os.path.basename(config_path)} ---", flush=True)

        config = load_config(config_path)

        if params_override:
            if 'solvers' in params_override and params_override['solvers']:
                if 'solvers' not in config: config['solvers'] = {}
                for key, value in params_override['solvers'].items():
                    if value is not None:
                        config['solvers'][key] = value

            if 'general' in params_override and params_override['general']:
                if 'general' not in config: config['general'] = {}
                for key, value in params_override['general'].items():
                    if value is not None:
                        config['general'][key] = value

        config_dir = os.path.dirname(config_path) if config_path else '.'

        task_file_path = config['task_params']['coordinates_file']
        if not os.path.isabs(task_file_path):
            task_file_path = os.path.abspath(os.path.join(config_dir, task_file_path))

        print(f"--- [子进程] 加载任务主文件: {os.path.basename(task_file_path)} ---", flush=True)
        task_data_df = pd.read_csv(task_file_path)

        num_runs = config.get('solvers', {}).get('llm_runs', 1)
        print(f"--- [子进程] 配置运行 {num_runs} 个任务组 ---", flush=True)

        all_results = []

        for group_id in range(1, num_runs + 1):
            print(f"\n{'='*15} [子进程] 开始运行任务组 {group_id}/{num_runs} {'='*15}", flush=True)

            row = task_data_df[task_data_df['Group'] == group_id]
            if row.empty:
                print(f"--- [子进程] 警告: 任务组ID {group_id} 在 {os.path.basename(task_file_path)} 中未找到. 跳过. ---", flush=True)
                continue

            task_coordinates = np.array(ast.literal_eval(row['task_coordinates'].iloc[0]))

            Q = np.array(config['robot_params']['q_matrix'])
            num_tasks = len(task_coordinates)
            Y = np.zeros((2, num_tasks)); Y[0, :] = [1 if i % 2 == 0 else 0 for i in range(num_tasks)]; Y[1, :] = 1 - Y[0, :]
            world_info = WorldInfo(limit_x=[0, config['general']['world_size']], limit_y=[0, config['general']['world_size']])
            agent_list, task_list = create_agents_and_tasks(Q, Y, world_info, config['robot_params']['start_coords'], task_coordinates.tolist())

            print(f"--- [子进程] 核心算法启动. SUSD迭代={config['solvers']['susd_iterations']}, LLM迭代={config['solvers']['llm_iterations']} ---", flush=True)

            results_one_run = run_llm_susd_flow(config, agent_list, task_list, run_group_id=group_id)

            print(f"--- [子进程] 任务组 {group_id} 核心算法完成. ---", flush=True)

            if isinstance(results_one_run, list) and results_one_run:
                all_results.extend(results_one_run)
            elif isinstance(results_one_run, str):
                print(f"--- [子进程] 任务组 {group_id} 未成功完成: {results_one_run} ---", flush=True)

        total_end_time = time.perf_counter()
        total_duration = total_end_time - total_start_time

        if all_results:
            for res in all_results:
                res['Total_Subprocess_Time'] = total_duration

            output_dir = os.path.join(project_root, "results")
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(config_path))[0]

            num_robots = config['general']['num_robots']
            file_name = f"{num_robots}R_MultiGroup_{base_name}_result.csv"
            output_path = os.path.join(output_dir, file_name)
            pd.DataFrame(all_results).to_csv(output_path, index=False)
            print(f"--- [子进程] 所有组运行完毕, 结果已保存.", flush=True)
            # This line is kept for potential debugging but is not used by the GUI anymore
            print(f"--- [子进程] ResultFile:{output_path} ---", flush=True)
        else:
            print(f"--- [子进程] 所有组运行完毕, 无有效结果可保存. ---", flush=True)

        return 0

    except Exception as e:
        print(f"--- [子进程] 发生严重错误 ---", flush=True)
        print(traceback.format_exc(), flush=True)
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single robot task allocation experiment.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--params_override', type=str, default='{}', help='Optional JSON string of parameter overrides, e.g. \'{"solvers": {"susd_iterations": 50}}\'. Defaults to no overrides.')

    args = parser.parse_args()

    params = json.loads(args.params_override)

    exit_code = run_single_task(args.config_path, params)
    sys.exit(exit_code)
