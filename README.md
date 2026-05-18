# Trae Reproduction Agent 使用说明

本仓库是在 Trae Agent 基础上改造的任务复现 agent。目标不是完整复现整篇论文，而是：给定一个仓库和一个 markdown 任务说明，让 agent 完成从环境配置、数据集下载、checkpoint/模型下载、命令执行、结果记录，到和 README/markdown 原始指标对比的闭环。

## 1. 前置条件

推荐在 Linux 服务器上运行。

需要准备：

- 已安装 Python 和 `uv`
- 已安装本仓库依赖
- 一个可访问的模型服务
- 一个目标代码仓库
- 目标仓库中有任务说明 markdown，例如 `task.md`

安装依赖：

```bash
cd /path/to/trae-agent
uv sync --all-extras
```

如果已经有 `.venv`，脚本会优先使用：

```bash
.venv/bin/trae-cli
```

否则会回退到：

```bash
uv run trae-cli
```

## 2. 模型服务统一配置

本项目现在把本地 vLLM、第三方 API 网关、OpenAI-compatible 服务统一成一种调用方式：

```text
provider = openai_compatible
endpoint = /v1/chat/completions
```

正常情况下你只需要改三个值：

```bash
BASE_URL=http://127.0.0.1:8000/v1
API_KEY=EMPTY
MODEL=Qwen3-8B
```

也就是说：

- 本地 vLLM：改 `BASE_URL` 和 `MODEL`。
- 第三方 API：改 `BASE_URL`、`API_KEY` 和 `MODEL`。
- 自建 OpenAI-compatible 网关：同样只改这三个值。

`PROVIDER` 默认就是 `openai_compatible`，一般不用改。

### 2.1 本地 vLLM 示例

如果你已经在 8000 端口启动了 vLLM，先检查：

```bash
curl http://127.0.0.1:8000/v1/models
```

如果返回模型名，例如：

```json
{
  "data": [
    {
      "id": "Qwen3-8B"
    }
  ]
}
```

后续 `MODEL` 就填 `Qwen3-8B`。

启动时：

```bash
./scripts/run_repro.sh --working-dir /path/to/target-repo
```

因为默认值已经是：

```bash
BASE_URL=http://127.0.0.1:8000/v1
API_KEY=EMPTY
MODEL=Qwen3-8B
PROVIDER=openai_compatible
```

### 2.2 第三方 API 示例

如果你换成第三方 API，只改参数：

```bash
./scripts/run_repro.sh \
  --working-dir /path/to/target-repo \
  --base-url https://your-api-host/v1 \
  --api-key your_api_key \
  --model your-model-name
```

或者用环境变量：

```bash
BASE_URL=https://your-api-host/v1 \
API_KEY=your_api_key \
MODEL=your-model-name \
./scripts/run_repro.sh --working-dir /path/to/target-repo
```

### 2.3 什么时候才改 provider

大多数情况不用改 `PROVIDER`。

只有当后端不是 `/v1/chat/completions`，而是 OpenAI Responses API 时，才使用：

```bash
--provider openai
```

如果你不确定，用默认的 `openai_compatible`。

## 3. 快速启动

准备目标仓库：

```bash
cd /path/to/target-repo
ls task.md
```

启动 agent：

```bash
cd /path/to/trae-agent
chmod +x scripts/run_repro.sh
./scripts/run_repro.sh --working-dir /path/to/target-repo
```

这个命令默认会让 agent 在目标仓库里读取 `task.md`，并执行任务复现流程。

如果目标 markdown 不叫 `task.md`，自定义任务提示：

```bash
./scripts/run_repro.sh \
  --working-dir /path/to/target-repo \
  --task "Read reproduce_gui_kv.md and reproduce exactly the task it describes. Do not use Docker."
```

## 4. 参数优先级

模型配置优先级：

```text
命令行参数 > 环境变量 > trae_config.repro.yaml.example
```

所以不需要手动改 YAML。常用改法就是启动时加：

```bash
--base-url ...
--api-key ...
--model ...
```

## 5. 直接使用 trae-cli

也可以不使用脚本，直接运行：

```bash
.venv/bin/trae-cli run "Read task.md and reproduce exactly the task it describes. Configure the environment, download required datasets and checkpoints, run the specified commands, extract reproduced metrics, extract original metrics from README or the markdown, and write results_comparison.md. Do not use Docker." \
  --agent-type env_setup_agent \
  --config-file trae_config.repro.yaml.example \
  --provider openai_compatible \
  --model Qwen3-8B \
  --model-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --working-dir /path/to/target-repo \
  --trajectory-file trajectories/task_repro.json \
  --max-steps 160
```

## 6. Agent 会做什么

`env_setup_agent` 当前执行的是“md 指定任务复现”流程：

1. 读取任务 markdown。
2. 阅读目标仓库 README、docs、requirements、pyproject、setup.py、scripts 等。
3. 从 README/markdown 中提取原始指标。
4. 生成 `repro_plan.md`。
5. 生成 `setup.sh`。
6. 生成 `download_assets.sh`，用于下载数据集、checkpoint、模型或样例输入。
7. 生成 `run_reproduction.sh`，用于执行任务指定命令。
8. 执行 setup、下载、复现命令。
9. 提取复现指标到 `.trae_env/reproduced_metrics.json`。
10. 提取原始指标到 `.trae_env/original_metrics.json`。
11. 生成 `results_comparison.md`，对比原始值、复现值、绝对差、相对差和波动分析。
12. 生成 `final_report.md`。

## 7. 输出产物

目标仓库中会生成：

```text
repro_plan.md
setup.sh
download_assets.sh
run_reproduction.sh
results_comparison.md
final_report.md
failure_analysis.md
.trae_env/
  commands.json
  original_metrics.json
  reproduced_metrics.json
  reproduction_verification.json
  repair_history.md
  logs/
```

`trajectory` 默认保存在 Trae Agent 仓库下：

```text
trajectories/task_repro.json
```

可以通过参数修改：

```bash
./scripts/run_repro.sh \
  --working-dir /path/to/target-repo \
  --trajectory-file trajectories/my_task.json
```

## 8. 失败处理

如果某个命令失败，agent 会收到：

- 命令返回码
- stdout/stderr 摘要
- 完整日志路径

完整日志保存在：

```text
.trae_env/logs/
```

prompt 会软约束模型先判断失败类型，再修复：

```text
environment
dependency version
dataset
checkpoint
command/entrypoint
metric_parse
unknown
```

更细的类别包括：

```text
python_version
dependency_too_new
dependency_too_old
dependency_conflict
pytorch_cuda
system_package
dataset_download
checkpoint_download
network
entrypoint
metric_parse
unknown
```

模型应把失败类别、证据和修复动作写入：

```text
.trae_env/repair_history.md
```

如果无法继续复现，应写：

```text
failure_analysis.md
```

环境问题的修复优先级：

1. 先看日志和 `pip freeze`/`pip show`/`pipdeptree`，判断包是缺失、太新、太旧，还是版本冲突。
2. 再看 README、requirements、pyproject、setup.py、lock 文件和 release 时间，推断兼容版本。
3. 优先修改 `setup.sh` 或环境文件，固定 Python、PyTorch/CUDA、pip 包、系统包版本。
4. 再检查数据集和 checkpoint 路径是否正确。
5. 不要一报错就修改仓库源码。只有确认是源码和文档运行时不兼容，且环境修复不可行时，才做最小源码修改，并在 `final_report.md` 里说明。

数据集和 checkpoint 规则：

- 数据集必须从 README/markdown/官方文档给出的地址下载，或从用户提供的本地路径复制。
- checkpoint / pretrained model 必须从文档来源下载，或从用户提供的本地路径复制。
- 不允许通过训练模型来代替 checkpoint 下载。
- 如果链接失效、权限不足或文件缺失，应在 `failure_analysis.md` 中记录 URL、错误信息、需要的文件名和期望路径。
- 除非 markdown 明确要求训练，否则训练出来的新权重不算有效复现 checkpoint。

## 9. 完成条件

agent 只有在以下条件满足后才允许结束：

- `bash run_reproduction.sh` 成功。
- `.trae_env/reproduction_verification.json` 存在且 returncode 为 0。
- `results_comparison.md` 已生成。

如果模型提前调用 `task_done`，框架会拒绝，并要求继续运行复现或写失败分析。

## 10. 安全边界

当前 bash 工具有基础安全过滤：

- 禁止 Docker 命令。
- 禁止危险 `rm -rf /`。
- 禁止 `sudo rm -rf`。
- 禁止破坏性 `git clean -fd`。
- 长输出会写入日志并截断返回，避免上下文爆炸。

本课题要求不使用 Docker；如果原仓库 README 只提供 Docker 路线，agent 应将其转写为 conda/venv/pip/shell 脚本方案，并在报告中说明等价关系。
