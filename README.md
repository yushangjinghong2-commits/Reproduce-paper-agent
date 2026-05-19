# Trae Reproduction Agent 使用说明

本仓库是在 Trae Agent 基础上改造的任务复现 agent。目标不是完整复现整篇论文，而是：给定一个仓库和一个你指定的复现目标提示词，让 agent 只阅读 `README.md`，从 README 中总结环境配置、数据集下载、checkpoint/模型下载、推理/评测命令、原始结果，然后复现这个目标并输出对比表。

## 1. 前置条件

推荐在 Linux 或 WSL 上运行。后续说明默认使用 Linux/WSL 的 `bash` 命令。

需要准备：

- 已安装 Python
- 已安装本仓库依赖
- 一个可访问的模型服务
- 一个目标代码仓库
- 目标仓库中有 `README.md`
- 你明确指定一个复现目标提示词，例如 `"复现 README 表 1 中 MuQ-MuLan 在 SongEval 上的 SRCC"` 或 `"跑 README 中的 zero-shot evaluation 命令并对比主结果"`

安装依赖：

```bash
cd /path/to/trae-agent
python -m venv venv
source venv/bin/activate
pip install -e .
```

脚本会优先使用当前激活的虚拟环境：

```bash
$VIRTUAL_ENV/bin/trae-cli
```

如果没有激活环境，再依次尝试：

```bash
.venv/bin/trae-cli
venv/bin/trae-cli
```

默认不会自动调用 `uv` 创建环境。如果你明确想让 `uv` 管理环境，可以使用：

```bash
USE_UV=1 ./scripts/run_repro.sh --working-dir ../target-repo --target "复现目标提示词"
```

这时才会回退到：

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
./scripts/run_repro.sh --working-dir ../target-repo --target "复现目标提示词"
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
  --working-dir ../target-repo \
  --target "复现目标提示词" \
  --base-url https://your-api-host/v1 \
  --api-key your_api_key \
  --model your-model-name
```

或者用环境变量：

```bash
BASE_URL=https://your-api-host/v1 \
API_KEY=your_api_key \
MODEL=your-model-name \
./scripts/run_repro.sh --working-dir ../target-repo --target "复现目标提示词"
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
cd ../target-repo
ls README.md
```

启动 agent：

```bash
cd /path/to/trae-agent
chmod +x scripts/run_repro.sh
./scripts/run_repro.sh --working-dir ../target-repo --target "复现目标提示词"
```

这个命令默认会让 agent 在目标仓库里只读取 `README.md`。读完后执行顺序是：先规划环境配置命令；再判断为了完成你的 `--target` 应该跑 README 里的哪条推理/评测命令；最后根据这条命令需要的输入，下载对应数据集、checkpoint、模型或样例输入。`--working-dir` 可以写相对路径；脚本会在内部解析成绝对路径传给 Trae CLI，但 agent 在目标仓库中生成的 `setup.sh`、`download_assets.sh`、`run_reproduction.sh` 会优先使用仓库相对路径。

如果你想补充额外要求，可以自定义任务提示，但默认约束仍然是只用 README 规划，并且只复现 `--target` 指定的目标：

```bash
./scripts/run_repro.sh \
  --working-dir ../target-repo \
  --target "复现目标提示词" \
  --task "Read only README.md and reproduce the specified target. Do not use Docker."
```

如果 `README.md` 不存在或无法读取，agent 应写 `failure_analysis.md` 说明缺失文件和证据。

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
.venv/bin/trae-cli run "Read only README.md and reproduce target prompt: 复现目标提示词. Configure the environment, download required datasets and checkpoints, run the README-documented inference/evaluation command, extract reproduced result, extract original result from README when present, and write results_comparison.md. Do not use Docker." \
  --agent-type env_setup_agent \
  --config-file trae_config.repro.yaml.example \
  --provider openai_compatible \
  --model Qwen3-8B \
  --model-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --working-dir ../target-repo \
  --trajectory-file trajectories/target-repo_$(date +%Y%m%d_%H%M%S).json \
  --max-steps 160
```

## 6. Agent 会做什么

`env_setup_agent` 当前执行的是“md 指定任务复现”流程：

1. 读取目标仓库 `README.md`，不读取其他仓库文件来规划任务。
2. 根据 README 先总结环境配置命令。
3. 根据 README 判断完成 `--target` 需要跑哪条推理/评测命令。
4. 根据这条命令反推需要哪些数据集、checkpoint、模型或样例输入。
5. 从 README 中提取这个目标对应的原始结果或参考值。
6. 生成 `repro_plan.md`，其中顺序为环境命令、目标运行命令、该命令所需资产、成功标准。
7. 生成 `setup.sh`。
8. 生成 `run_reproduction.sh`，用于执行目标命令。
9. 生成 `download_assets.sh`，只下载目标命令需要的数据集、checkpoint、模型或样例输入。
10. 依次执行 setup、下载、复现命令。
11. 提取复现结果到 `.trae_env/reproduced_metrics.json`。
12. 提取原始结果到 `.trae_env/original_metrics.json`。
13. 生成 `results_comparison.md`，只对比这个目标的原始结果、复现结果、数值差异和波动分析。
14. 生成 `final_report.md`。

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
trajectories/<repo>_<YYYYmmdd_HHMMSS>.json
```

可以通过参数修改：

```bash
./scripts/run_repro.sh \
  --working-dir ../target-repo \
  --target "复现目标提示词" \
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
2. 再看 README 中的安装命令、版本说明和当前 `pip freeze`/`python --version`/`torch`/`cuda` 信息，推断兼容版本。
3. 创建 conda 环境时必须指定 Python 版本；README 写明版本就使用 README 版本，没写时默认 `python=3.12`。
4. README 没写 PyTorch 版本时，优先尝试 `torch==2.6.*`、`torchvision==0.21.*`、`torchaudio==2.6.*`，CUDA 环境优先按 CUDA 12.4 wheel 配置。
5. README 没写 Transformers 版本时，优先尝试 `transformers==4.55.*`。
6. 包太新就降级，包太旧就升级，CUDA/torch 不匹配就换匹配的 PyTorch/CUDA 组合。
7. 每次修复都记录到 `.trae_env/repair_history.md`，并重新执行失败阶段；不能第一次失败就写 `failure_analysis.md`。
8. 再检查数据集和 checkpoint 路径是否正确。
9. 不要一报错就修改仓库源码。只有确认是源码和文档运行时不兼容，且环境修复不可行时，才做最小源码修改，并在 `final_report.md` 里说明。

数据集和 checkpoint 规则：

- 数据集必须从 README 给出的地址下载，或从用户提供的本地路径复制。
- checkpoint / pretrained model 必须从 README 来源下载，或从用户提供的本地路径复制。
- 不允许通过训练模型来代替 checkpoint 下载。
- 如果链接失效、权限不足或文件缺失，应在 `failure_analysis.md` 中记录 URL、错误信息、需要的文件名和期望路径。
- 除非 README 明确要求训练，否则训练出来的新权重不算有效复现 checkpoint。

## 9. 完成条件

agent 只有在以下条件满足后才允许结束：

- `bash run_reproduction.sh` 成功。
- `.trae_env/reproduction_verification.json` 存在且 returncode 为 0。
- `.trae_env/reproduced_metrics.json` 包含真实执行得到的非空复现结果，不能是 `blocked`、`failed`、`not_run`、占位符或计划值。
- `results_comparison.md` 已生成。

如果模型提前调用 `task_done`，或者只在文本里说“我要 task_done”但没有真实指标，框架会拒绝并把本次运行标记为失败，避免在无效完成状态里循环。

## 10. 安全边界

当前 bash 工具有基础安全过滤：

- 禁止 Docker 命令。
- 禁止危险 `rm -rf /`。
- 禁止 `sudo rm -rf`。
- 禁止破坏性 `git clean -fd`。
- 长输出会写入日志并截断返回，避免上下文爆炸。
- `setup.sh`、`download_assets.sh`、`run_reproduction.sh`、`pip install`、`conda create/install`、`wget/curl`、`huggingface-cli download` 等长命令会自动作为后台 job 运行，立即返回 `job_id`。
- 后台 job 没有固定时间上限；agent 应继续用 bash 工具传入 `job_id` 轮询进度，直到日志显示 job 结束。
- 轮询会刷新显示 `.trae_env/logs/job_xxxx.log` 的尾部内容，用于观察安装、下载、推理或评测进度。
- 如果人判断任务卡住或方向错误，可以用同一个 `job_id` 加 `kill=true` 终止该后台任务。
- 前台命令默认不设置固定超时；如果确实需要限制，可用环境变量 `TRAE_BASH_TIMEOUT` 或 bash 工具参数 `timeout` 设置。

本课题要求不使用 Docker；如果原仓库 README 只提供 Docker 路线，agent 应将其转写为 conda/venv/pip/shell 脚本方案，并在报告中说明等价关系。
