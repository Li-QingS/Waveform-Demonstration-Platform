# FDIDM / OFDM / OTFS / AFDM 波形演示平台

基于 Python、PyQt5 和 pyqtgraph 的桌面实验平台，提供四种波形的本地仿真、星地链路参数研究、三波形配对比较，以及 USRP 硬件链路验证。平台 AI 助手可结合当前页面、图表和日志解释结果、提供操作指导。

纯仿真无需 USRP、GNU Radio 或 AI 服务。硬件验证和 AI 对话分别按需配置。

## 安装与启动

项目声明支持 **Python 3.10 及以上版本**；仓库的 CI 和 conda 环境使用 **Python 3.11**。以下命令在仓库根目录执行。

Windows PowerShell 本地仿真环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m waveform_sim.main
```

Linux / macOS 可在创建环境后用 `source .venv/bin/activate` 激活，再执行相同的安装和启动命令。依赖版本见 [requirements.txt](requirements.txt) 和 [pyproject.toml](pyproject.toml)。后文的 `python` 均指所选环境中的解释器。

## 页面与使用流程

启动后从首页选择“仿真”或“硬件验证”，再在工作区内选择页面。页面首次访问时加载；先配置参数，再手动启动任务。

| 工作区 / 页面 | 主要能力 |
| --- | --- |
| 仿真 / FDIDM 软波形 | α/β 参数、理论 SER 搜索、预测式自适应、性能面、星座与频谱、参数轨迹和时变信道观察 |
| 仿真 / OFDM | 实时星地链路；Eb/N0、残余 CFO、子载波间隔、时延缩放等性能研究；JSON 导出 |
| 仿真 / OTFS | 实时星地链路与 DD 能量显示；增加多普勒扩展、时隙数等研究维度；JSON 导出 |
| 仿真 / AFDM | 实时星地链路、c1/c2、CPP/CP 和有效 DAFT 信道；参数扫描与 JSON 导出 |
| 仿真 / 波形对比 | OFDM / OTFS / AFDM 实时配对比较与 Monte-Carlo 自动扫描；统一资源、载荷、物理信道和噪声；JSON 导出 |
| 硬件 / FDIDM 自适应验证 | GMI 有限波形库选择、A-B-B-A 真链路验证、α/β 性能面、链路诊断、观测窗口及日志导出 |
| 硬件 / 三波形链路测评 | 分别选择 OFDM / OTFS / AFDM，连接 USRP、发送文本、查看恢复文本、CRC、误码指标及图表 |

顶部可返回首页或切换工作区；`Alt+Home` 返回首页，`Ctrl+Q` 退出。离开功能页会停止当前任务，保留参数和最近结果；硬件连接会释放，返回后需重新连接并启动。

### FDIDM 的两种自适应路径

**仿真侧**依据 CSI 快照进行预测 SER 粗细搜索，结合收益门限、连续推荐稳定性和冷却时间更新 α/β。可选择固定信道、动态块衰落、帧内快时变和连续多普勒，观察参数与性能随时间的变化。

**硬件侧**使用接收机相关 GMI 对有限波形库评分。库预算可选 4 / 16 / 36，接收机可选对角 K=0 或完整匹配对照，并提供快速、标准、保守鲁棒等级。GMI 按包含数据、保护、训练和控制开销的共同资源归一化，单位为 **nats / charged sample**；选择考虑不确定性和切换代价。默认关闭自适应，启用后，满足门限的候选进入 A-B-B-A 非回归验证。GMI 负责选择，EVM / SER / CRC 提供实际链路证据；候选得分提高本身不等于实测改善。完整匹配对照用于验证该模型下 GMI 对 α/β 的不变性。

### 结果的适用范围

OFDM / OTFS / AFDM 独立研究页采用 NTN 时延功率轮廓与简化多径、多普勒模型，并提供训练或导频估计与理想 CSI 参考。三页的训练、前缀和接收机预算有所不同，曲线不能直接拼成同资源排名。

波形对比页使用独立的共同资源模型，默认三波形均为理想 CSI 完整 LMMSE；“差异接收机演示”包含接收机复杂度差异。研究与对比结果保留真实误码计数；零误码仍为 0，对数图另用标记显示 95% Wilson 上界。结果受信道、种子、帧数、同步与接收机假设限制，不保证某种波形普遍占优，也不代表完整标准一致性测试或星地链路预算。

## 可选：Windows USRP 硬件环境

[environment.yml](environment.yml) 提供 Python 3.11、GNU Radio 3.10、GNU Radio UHD、UHD 4.x 和 Qt 等依赖：

```powershell
conda env create -f environment.yml
conda activate waveform-demo-platform
python scripts/check_environment.py
python scripts/check_usrp.py
python -m waveform_sim.main
```

硬件验证需在同一运行环境中具备可导入的 `gnuradio` / `gnuradio.uhd`、匹配的 UHD 驱动与设备镜像，以及可被 UHD 发现的设备。检查脚本报告 `PASS` / `WARN`；纯仿真环境缺少硬件组件时出现 `WARN` 不影响仿真。

当前专用验证和 RF 端口诊断脚本面向 **USRP B210**。实际测试前应核对自己的设备序列号、TX/RX 通道与天线端口、中心频率、采样率和增益；有线回环需配置适当外部衰减，OTA 需使用合适的天线和发射条件。先连接 / 配置设备，再启动测试；设备可发现不代表链路已完成验证。其他 USRP 型号需单独适配与核验。

## 可选：平台 AI 助手

首页和全部实验页共用右下角 **AI** 悬浮入口，`Ctrl+J` 打开助手。对话窗口可与平台同时操作，切页保留可见对话、草稿和缩放；新问题读取当前页面快照，旧回答保留提问时的来源。

- 快捷入口：日志分析、运行结果图解释、异常原因排查、优化建议、操作指导。
- 输入：`Enter` 发送，`Shift+Enter` 换行；可复制或保存对话。
- 缩放：标题栏按钮、`Ctrl+滚轮` 或 `Ctrl++` / `Ctrl+-`，范围 80%–140%；`Ctrl+0` 恢复 100%。
- 数据来源：当前页图表、参数、运行状态和日志，也可选择历史 `.log` / `.txt`。历史日志支持 UTF-8 / GB18030，文件上限 16 MiB，发送的日志正文最多保留末尾 24,000 个字符。

只有发送问题或点击快捷入口时才会向配置的分析服务提交相关上下文。原始日志作为后台证据，不加入复制或保存的对话。助手提供解释与建议，不代替用户连接设备、启动实验或修改参数。取消会忽略未返回结果，已经发出的服务端请求可能继续处理。

### 管理员配置

助手通过 [ai_config.py](waveform_sim/service/ai_config.py) 加载后台配置，调用 DeepSeek 兼容的聊天接口。环境变量优先于本地保存值：

| 环境变量 | 用途 / 默认值 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 管理员注入的 API 凭据；必需，无内置密钥 |
| `DEEPSEEK_BASE_URL` | API 基础地址；默认 `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 模型名称；代码默认 `deepseek-flash`，应按服务端实际可用模型配置 |

Windows 本地部署也可调用 `save_ai_config()`，将配置以当前 Windows 用户绑定的 DPAPI 数据保存到 `%LOCALAPPDATA%\WaveformDemonstrationPlatform\ai-config.dpapi`。请在运行平台的同一 Windows 用户下执行。先运行 `python` 进入交互式解释器，再执行以下代码；密钥以隐藏方式输入，不写进命令文本：

```python
from getpass import getpass
from waveform_sim.service.ai_config import (
    DEFAULT_AI_BASE_URL, DEFAULT_AI_MODEL, save_ai_config,
)
from waveform_sim.service.ai_log_analyzer import DeepSeekConfig

base_url = input("API base URL: ").strip() or DEFAULT_AI_BASE_URL
model = input("Model: ").strip() or DEFAULT_AI_MODEL
save_ai_config(DeepSeekConfig(base_url, model, getpass("API Key: ")))
```

环境变量方式可用于管理员管理的运行环境和 CI，也适用于不使用 Windows DPAPI 的部署。源码不读取 `.env` 文件；变量须由启动平台的进程继承。共享部署应由服务端保管密钥，向客户端提供受控接口。不要把实际密钥提交到仓库或写入实验日志。

## 命令行工具

以下工具在仓库根目录运行；带参数的脚本可用 `--help` 查看完整选项。

| 命令 | 用途与输出 |
| --- | --- |
| `python scripts/check_environment.py` | Python / Qt / GNU Radio / UHD 环境与 USRP 探测 |
| `python scripts/check_usrp.py` | 通过 UHD 枚举设备 |
| `python scripts/run_self_test.py` | 无显示、无射频的 FDIDM 出帧与实验记录自检；临时产物在退出后清理 |
| `python scripts/study_ofdm_ntn.py --frames 40 --seed 42` | OFDM 研究；默认 `output/ofdm-ntn-study.json` |
| `python scripts/study_otfs_ntn.py --frames 40 --seed 42` | OTFS 研究；默认 `output/otfs-ntn-study.json` |
| `python scripts/study_afdm_ntn.py --frames 20 --seed 42` | AFDM 研究；默认 `output/afdm-ntn-study.json` |
| `python scripts/reproduce_gmi_paper_example.py --json artifacts/gmi-paper-example.json` | GMI 有限库数值示例与图；默认图为 `artifacts/gmi_paper_example.png`，`--no-figure` 只计算数值 |
| `python scripts/validate_alpha_beta_adaptation.py --output output/alpha-beta-validation.json` | 离线校验预测 SER 优化器；用于诊断与回归，硬件在线切换由 GMI 控制 |
| `python scripts/validate_gmi_hardware.py --help` | B210 固定 / 对角 GMI / 完整匹配验证；实测需显式提供 `--serial`、`--tx-gain`、`--rx-gain`，默认报告为 `log/gmi_hardware_validation.json` |
| `python scripts/diagnose_b210_rf_ports.py --help` | B210 RF 端口与增益扫描；实际运行会发射测试音，应显式指定自己的 `--serial` 与增益方案；结果逐条输出 JSON |

三种 NTN 研究脚本均支持 `--output` 自定义 JSON 路径。`--help` 只显示参数；硬件脚本的实际运行需要完成设备与射频链路配置。

## 日志与实验产物

- FDIDM 硬件页的“导出日志”默认保存为 `log/fdidm_debug_时间戳.log`；三波形硬件页在“测试日志”中显示运行记录。
- 独立研究页与波形对比页通过导出按钮保存 JSON，包含运行配置、原始计数和研究结果；保存路径由用户选择。
- 使用 `ExperimentService` 或 `LinkSimulator(auto_start_run=True)` 时，实验服务默认创建 `runs/<run_id>/`，可记录 `config.json`、`events.jsonl`、`metrics.csv`、`figures/`，结束时写入 `report.md`。GUI 各页并非统一自动接入该记录流程。
- `output/`、`artifacts/`、`runs/` 和运行日志属于本地产物，不作为源码版本内容。

## 测试

安装 `requirements.txt` 后，可运行包含 Qt 界面用例的本地回归测试。Windows PowerShell 无显示模式：

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:MPLBACKEND = 'Agg'
python scripts/run_self_test.py
python -m pytest -q
python -m compileall waveform_sim
```

测试覆盖数值内核、NTN 模型、配对比较、GMI、硬件抽象与接收链路、实验记录、导航及 AI 上下文等；使用模拟对象的硬件测试不能代替 B210 实测。

[GitHub Actions CI](.github/workflows/ci.yml) 使用 Ubuntu、Python 3.11 和 `MPLBACKEND=Agg`，只安装 NumPy、Matplotlib、pytest 后运行 pytest 与 compileall。未安装 Qt / pyqtgraph 时相应 UI 用例自动跳过；因此 CI 不等于完整桌面界面验证，也不包含真实射频收发。

## 目录与技术参考

```text
waveform_sim/
  main.py          # 桌面入口：python -m waveform_sim.main
  core/            # 配置、调制、变换、指标和统一链路引擎
  simulation/      # FDIDM 自适应、NTN 研究、共同资源比较后端
  gmi/             # 资源契约、有限库、接收机相关评分与选择
  hardware/        # USRP 收发、同步、FEC、CSI、自适应及证据统计
  service/         # 实验记录、事件日志、AI 配置与分析服务
  diagnostics/     # 环境检查、快照和报告导出
  ui/              # 首页、工作区、实验页与全局 AI 助手
scripts/           # 可重复运行的研究、诊断与验证入口
tests/             # 数值、服务、硬件抽象和 Qt 回归测试
markdown/          # NTN 模型分析与波形对比技术说明
.github/workflows/ # 核心 CI
```

- [OFDM 星地链路分析](markdown/OFDM_NTN_ANALYSIS.md)
- [OTFS 星地链路分析](markdown/OTFS_NTN_ANALYSIS.md)
- [AFDM 星地链路分析](markdown/AFDM_NTN_ANALYSIS.md)
- [波形对比模型、比较契约与审计](markdown/WAVEFORM_COMPARE_AUDIT.md)
