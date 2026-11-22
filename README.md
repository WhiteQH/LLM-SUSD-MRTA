# Agentic Task Allocation for Heterogeneous Multi-Robot Systems: A Hybrid LLM and Local Search Approach

This repository provides a **reference implementation** of an agentic hybrid
framework that combines a Large Language Model (LLM) with local search for
heterogeneous multi-robot task allocation (MRTA), illustrated with an AUV
survey/inspection-style example.

> **About This Repository / 关于本仓库**
>
> - This repository provides a reference implementation of the hybrid LLM and
>   local search framework for heterogeneous multi-robot task allocation: an
>   LLM acts as an agentic allocation initializer that generates informed
>   candidate allocations, and the Speed-Up Slow-Down (SUSD) derivative-free
>   local search optimizer refines them numerically, with an external memory
>   feeding evaluated records back into subsequent prompts. It covers the
>   complete methodological pipeline — LLM initial allocation → route
>   evaluation → SUSD refinement → memory-guided iterative feedback — and
>   serves as a methodological and engineering reference for related research.
> - 本仓库提供混合 LLM 与局部搜索框架（异构多机器人任务分配）的参考实现：
>   LLM 作为智能体生成初始任务分配，Speed-Up Slow-Down（SUSD）无导数局部
>   搜索优化器对其进行数值精炼，外部记忆将已评估的历史记录注入后续提示词。
>   涵盖 LLM 初始分配 → 路径评估 → SUSD 精炼 → 记忆引导迭代反馈的完整
>   方法论流程，可作为相关研究的方法论与工程架构参考。

## Method Overview

Following the paper's methodology, the core pipeline (`src/llm_susd_module.py`)
runs an iterative agentic optimization loop coupling LLM-based candidate
generation, SUSD-based derivative-free refinement, makespan evaluation, and
external-memory update. Here *agentic* refers to this closed-loop interaction
among the components; the LLM does not control robot execution.

1. **LLM-based allocation initialization** — a structured prompt describing
   robot capabilities, task types and coordinates is sent to an Azure OpenAI
   deployment; the response is parsed and structurally validated (every task
   assigned exactly once, no robot left empty), retrying up to 10 times
   (`llm_max_retries`) on invalid or unparsable responses.
2. **Route evaluation** — a nearest-neighbor greedy route is computed per robot
   and the makespan of the allocation is evaluated (`src/path_planning.py`,
   `src/greedy_validation.py`).
3. **SUSD refinement** — the validated allocation seeds the Speed-Up Slow-Down
   (SUSD) algorithm, a derivative-free local search optimizer that
   stochastically refines the task bundles toward a lower makespan.
4. **Memory-guided feedback** — the initial allocation, the SUSD-refined
   allocation and the resulting makespan are stored as a memory record and
   injected into the next iteration's prompt as feedback-conditioned context.
   The memory keeps the most recent records (FIFO, bounded by
   `llm_max_context`), adapting subsequent proposals without updating LLM
   parameters.

The loop repeats for `llm_iterations` outer iterations; the allocations and
makespans of every iteration are recorded in the result CSV.

```
LLM allocation (A1) → greedy route evaluation (A2) → SUSD refinement (A3)
        ↑                                               ↓
        └──── memory-guided feedback ← re-planning + makespan (A4)
```

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── run_task.py                  # Headless CLI entry point (no GUI dependencies)
├── src/
│   ├── __init__.py
│   ├── config_manager.py        # YAML config load/save
│   ├── data_management.py       # Agent / Task / WorldInfo dataclasses
│   ├── path_planning.py         # Nearest-neighbor routing & bundle exec time
│   ├── greedy_validation.py     # Greedy re-planning & makespan evaluation
│   └── llm_susd_module.py       # Core agentic LLM + SUSD hybrid loop
├── configs/
│   └── example.yaml             # 4-robot / 20-task example configuration
└── data/
    ├── example_4R20T.csv        # Synthetic example coordinates
    └── TaskGenerate.ipynb       # Generator script for synthetic coordinate CSVs
```

## Installation

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

> **Important — openai version:** the code uses the *legacy* OpenAI Python
> interface (`openai.ChatCompletion.create` with `api_type="azure"`), which is
> specific to the 0.28.x SDK. Use `openai==0.28.1` as pinned in
> `requirements.txt`; the 1.x SDK uses a different API and is not compatible.

## API Key Setup

The LLM module runs with your own Azure OpenAI credentials. Access to the Azure
OpenAI deployment used in our experiments can be applied for through **HKUST
internal systems**; alternatively, you can use your own Azure OpenAI resource.

Configure credentials in either of two ways (environment variables take
precedence over the YAML values):

1. **Environment variables**

   ```bash
   # Linux / macOS
   export AZURE_OPENAI_KEY="<your-key>"
   export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
   ```

   ```powershell
   # Windows PowerShell
   $env:AZURE_OPENAI_KEY = "<your-key>"
   $env:AZURE_OPENAI_ENDPOINT = "https://<your-resource>.openai.azure.com/"
   ```

2. **Edit `configs/example.yaml`** — replace the placeholders under `api_keys`:

   ```yaml
   api_keys:
     azure_openai_key: YOUR_AZURE_OPENAI_KEY_HERE
     azure_openai_endpoint: https://YOUR-RESOURCE.openai.azure.com/
     azure_deployment_name: gpt-4o   # set to your own deployment name
   ```

## Running the Example

```bash
python run_task.py --config_path configs/example.yaml
```

Optional parameter overrides (JSON string, merged over the YAML config):

```bash
python run_task.py --config_path configs/example.yaml \
  --params_override '{"solvers": {"susd_iterations": 50, "llm_iterations": 1}}'
```

- `--config_path` (required): path to a YAML configuration file.
- `--params_override` (optional, default `{}`): JSON with `solvers` / `general`
  sections whose non-null values override the config.

The runner reads task groups (`Group`, `task_coordinates` columns) from the CSV
referenced in the config, executes the LLM+SUSD loop for each group, and writes
a result CSV to `results/` (git-ignored). LLM requests use the Azure OpenAI
credentials configured in [API Key Setup](#api-key-setup).

## Data Statement

All coordinates under `data/` are **programmatically synthesized example data**
(uniform random points, see `data/TaskGenerate.ipynb`). They serve as an
input-format reference and are separate from the experimental data used in the
paper.

CSV format: two columns — `Group` (task-group id, 1-based) and
`task_coordinates` (a stringified Python list of `[x, y]` pairs).

## Citation

If you find this architecture useful, please cite our paper *Agentic Task
Allocation for Heterogeneous Multi-Robot Systems: A Hybrid LLM and Local Search
Approach* (citation details will be added upon publication).

## License

Released for academic and educational purposes.