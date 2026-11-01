# Completion-Sound Performance Monitoring

论文 **Completion-Sound Performance Monitoring under Out-of-Order Delayed Labels: Finite-Window Confirmation and Complete Event-Time Cohorts** 的研究代码。

[English](README.md) | 简体中文

框架将有限窗口损失补全判定、模型版本权限和完整事件时间训练队列结合，用于延迟、乱序反馈下的性能监控与选择性自适应。

## 1. 配置环境

使用 **Python 3.12**。参考 CPU 环境为 Python 3.12.14、NumPy 2.3.5、pandas 2.2.3、SciPy 1.17.0、scikit-learn 1.8.0 和 joblib 1.5.3。

打开同时包含 `README.md`、`requirements.txt`、`scripts` 和 `data` 的仓库根目录。Windows 用户在文件夹地址栏输入 `cmd`，按回车，然后逐行执行：

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Linux/macOS：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

以后打开新终端时，回到仓库根目录，重新激活 `.venv` 即可。

额外依赖按任务放在 `environments/`：`generation.txt` 包含 River、Matplotlib、tqdm，`bts.txt` 包含 tzdata。ARF 训练使用兼容 [scikit-multiflow](https://scikit-multiflow.github.io/) 的独立环境。第 4 节基础复现使用固定版本的 CPU 核心依赖，进一步说明见 [docs/environment.md](docs/environment.md)。

## 2. 目录说明

| 路径 | 主要内容 |
| --- | --- |
| `src/completion_monitoring/` | 监控引擎、延迟反馈、模型版本权限和队列更新。 |
| `scripts/` | 文件检查、统计复算和实验启动入口。 |
| `experiments/` | 合成流、基线、协议审计、统计和 Retailrocket 实验。 |
| `data/frozen/` | 105 条合成实现：5 条固定流、50 条 LR/DT 流、50 条 ARF 流。 |
| `data/raw/` | Electricity、Weather 和筛选后的 BTS 输入。 |
| `data/processed/retailrocket/` | 固定模型输入、标签释放安排和任务定义。 |
| `data/statistics/`、`data/reference/` | 统计源评分和存档实验结果。 |
| `configs/`、`environments/` | 实验入口、参数说明和可选依赖。 |
| `extensions/`、`data/extensions/` | BTS/Freddie Mac 扩展脚本和存档记录。 |
| `tests/`、`third_party/`、`docs/` | 补全规则测试、上游生成器代码和详细说明。 |
| `results/` | 新运行生成的结果与工作副本。 |

保留这些路径层级。`configs/protocol.json` 记录参数说明，实际执行常量仍在对应驱动中。

## 3. 数据来源

研究涉及五个真实数据来源和五类合成流，其中 BTS、Freddie Mac 为扩展研究。

| 真实数据来源 | 获取网站 | 包内材料 |
| --- | --- | --- |
| Electricity | [基准仓库](https://github.com/FabianHinder/Creating-Drift-Benchmarks) · [CSV 下载](https://raw.githubusercontent.com/FabianHinder/Creating-Drift-Benchmarks/main/stream_ds/electricity.csv.xz) | `data/raw/benchmarks/` 下的压缩 CSV。 |
| Nebraska Weather | [原始基准页面](https://users.rowan.edu/~polikar/nse.html) · [CSV 下载](https://raw.githubusercontent.com/FabianHinder/Creating-Drift-Benchmarks/main/stream_ds/NEweather.csv.xz) | 包内文件为 `data/raw/benchmarks/neweather.csv.xz`。 |
| Retailrocket | [Kaggle 数据集](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) | 处理后留出集：10,594 行、24 个特征、10,194 条在线标签释放记录。 |
| BTS 航班准点数据 | [官方入口](https://www.transtats.bts.gov/ONTIME/) · [数据下载表](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=Nv4%20Pn44vr45) | 2024 年 1–3 月的三个 ATL 出发航班筛选 CSV。 |
| Freddie Mac 房贷数据 | [官方数据与文档](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset) · [下载门户](https://claritydownload.fmapps.freddiemac.com/) | 派生研究记录；原始 TXT 通过提供方获取。 |

| 合成流类型 | 官方生成器页面 |
| --- | --- |
| Agrawal | <https://riverml.xyz/latest/api/datasets/synth/Agrawal/> |
| Mixed | <https://riverml.xyz/latest/api/datasets/synth/Mixed/> |
| RandomTree | <https://riverml.xyz/latest/api/datasets/synth/RandomTree/> |
| SEA | <https://riverml.xyz/latest/api/datasets/synth/SEA/> |
| Sine | <https://riverml.xyz/latest/api/datasets/synth/Sine/> |

漂移构造使用 [Creating-Drift-Benchmarks](https://github.com/FabianHinder/Creating-Drift-Benchmarks)。105 个冻结文件是这五类生成器的不同实现。第 4 节的基础命令直接读取包内输入。

原始数据重建使用包含 `events.csv` 的 Retailrocket ZIP，或 Freddie Mac 2017–2022 年样本放款与表现文件。获取及预处理说明见 [docs/data.md](docs/data.md) 和 [docs/experiments.md](docs/experiments.md)。

## 4. 运行与结果

在仓库根目录、已激活的环境中，按顺序执行：

```bash
python scripts/verify_integrity.py
python -m unittest discover -s tests -v
python scripts/check_protocol.py
python scripts/reproduce_statistics.py
python experiments/real_world/replay_retailrocket.py
```

| 命令或脚本 | 得到什么 | 在哪里查看 |
| --- | --- | --- |
| `verify_integrity.py` | 检查 576 个数据文件、47 个 Python 文件和冻结流数量。 | 终端显示 `"passed": true`。 |
| `unittest` | 对大小 1–6 的窗口进行补全规则穷举测试。 | 终端显示 `OK`。 |
| `check_protocol.py` | 在一条冻结 SEA 流上运行 LR/DT，各四种策略，共 8 个结果。 | `results/smoke/metrics.csv` 和 `verification.json`。 |
| `reproduce_statistics.py` | 表 8/S1 的配对差值、秩、有效样本量和检验结果。 | `results/statistics/table8/`、`tableS1/` 和 `verification.json`。 |
| `replay_retailrocket.py` | 固定时间留出集上的四种 DT 策略。 | `results/retailrocket/metrics.csv`；验证文件显示 `"matches_reference": true`。 |

表 8 的 `waitall_accuracy_50_stream_differences.csv` 保存全部 50 条流差值，`waitall_accuracy_14_nonzero_ranks.csv` 保存 14 条非零差值的秩。统计定义见 [docs/statistics.md](docs/statistics.md)。

## 5. 其他实验

先列出入口，再选择运行：

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py protocol_ablation --smoke --jobs 1
python scripts/run_experiment.py preparation
python scripts/run_experiment.py coverage --smoke --jobs 1
```

`protocol_ablation` 检查未决窗口及更新权限；`preparation` 从已有配对窗口记录计算准备重叠；`coverage` 计算监控覆盖。`--smoke` 选择小规模运行，`--jobs` 设置并行进程数。驱动输出通常在 `results/实验名/work/result/`，完整入口说明见 [docs/experiments.md](docs/experiments.md)。

再次运行时使用新的输出目录，例如：

```bash
python scripts/check_protocol.py --output results/smoke_second
```

新计算写入 `results/`，分发的参考记录保留在 `data/reference/`。扩展入口状态：BTS 预处理可运行，下游回放需要适配源目录相对路径；Freddie Mac 回放需要提供方原始文件。

## 6. 常见问题与文档

| 情况 | 处理方法 |
| --- | --- |
| 找不到脚本或 requirements.txt。 | 在同时包含 README、scripts 和 data 的仓库根目录执行。 |
| 提示已有 work 目录。 | 用 `--output` 指定新的输出目录。 |
| 缺少生成器或 ARF 依赖。 | 按 `environments/` 的任务分组配置对应环境。 |

详细说明：[实验](docs/experiments.md) · [数据](docs/data.md) · [统计](docs/statistics.md) · [环境](docs/environment.md)。

## 7. 许可

项目软件采用 [MIT 许可](LICENSE)。上游代码保留原许可；数据使用遵循提供方条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
