# FDIDM 软波形自适应演示平台

基于论文 *Fractional Dual Index Division Multiplexing: A Soft Waveform Design Toward Integrated Satellite-Terrestrial Networks* 开发的波形仿真与 USRP 硬件验证平台，支持 FDIDM / OFDM / OTFS / AFDM 四种波形的软件仿真、性能对比和 USRP B210 真机链路测试。

## 核心能力

- 四种波形（FDIDM / OFDM / OTFS / AFDM）软件仿真；OFDM / OTFS / AFDM 提供独立星地研究引擎，三波形对比采用配对 Monte-Carlo 扫描
- USRP B210（N210 / X310）硬件链路测试与 RF 安全保护
- FDIDM α/β 预测式自适应闭环：硬件侧（`hardware/fdidm_hardtest.py`）与仿真侧（`simulation/fdidm_adaptive.py`）均实现“CSI 快照 → 理论 SER 搜索 → 稳定判定 → 冷却 → 自动应用”
- FDIDM 仿真页自适应过程可视化：α/β 轨迹、预测 SER 对比、切换标注、状态文本
- 时变信道模型（仿真页可切换，详见下文）：固定信道 / 动态块衰落 / 帧内快时变 / 连续多普勒
- 仿真图右下角“实时性能随时间变化”图：固定滚动时间窗、Y 轴按可见数据自动拟合、3 s 滑动平均、增量刷新
- 自适应过程页 SER-SNR 对比图：固定用户种子与信道实现，每个 SNR 点保持相同信道条件，观察 FDIDM 参数和自适应行为
- 工程化：统一配置、统一引擎、实验 artifact + JSONL 事件日志、诊断脚本、行为基线测试与 CI

## 时变信道模式

FDIDM 仿真页“时变模式”下拉提供四种信道演化方式，参数基于真实 LEO 星地链路（3GPP TR 38.811）：

- **固定信道**：整个运行期间保持同一 H_TF。
- **动态块衰落**：路径表（时延/幅度/多普勒）固定，每个相干块对路径相位做一次均值回归（AR(1)）小幅游走；相邻块信道相关，SER 随时间平稳变化而不是独立跳变。
- **帧内快时变**：每帧重新生成信道实现，并在帧内符号间做 AR(1) 增益演化。
- **连续多普勒**：按各径真实多普勒逐帧旋转路径相位 exp(j2π·fD·T_frame)，信道连续演化、种子不变、可复现。

参考量级：LEO 轨道速度约 7.8 km/s（28080 km/h）；20 GHz 载频下最大多普勒约 ±480 kHz、多普勒变化率可达 −5.44 kHz/s；默认径向系数 0.10 对应约 52 kHz 残余多普勒，此时相干时间仅数微秒——信道在帧级尺度上变化很快是物理事实，演示页用上述模式分别展示慢变/快变场景。

## 目录结构

```text
waveform_sim/
  main.py                       # 程序入口（PyQt5 主窗口）
  core/                         # 工程化内核
    config.py                   #   统一配置（Waveform/Adaptive/Hardware/Experiment）
    engine.py                   #   统一链路引擎 LinkSimulator
    modem.py / transforms.py / waveforms.py / metrics.py
  simulation/                   # 软件仿真后端（transceiver 兼容壳）
    simple_fdidm_rx.py / simple_ofdm_rx.py / simple_otfs_rx.py / simple_afdm_rx.py
    fdidm_adaptive.py           #   FDIDM 仿真侧预测式自适应（mixin + 搜索内核）
    compare_scan_backend.py     #   三波形 Monte-Carlo 对比扫描
    ofdm_ntn.py / otfs_ntn.py / afdm_ntn.py  # 独立星地实时链路与可复现研究
  hardware/                     # 硬件后端（兼容壳 + 抽象层）
    fdidm_hardtest.py / ofdm_hardtest.py / otfs_hardtest.py / afdm_hardtest.py
    transport.py / rf_safety.py / device_manager.py / iq_replay.py
    stream.py / channel.py / fec.py / gr_flow.py / fdidm_adaptive.py
  service/                      # 实验记录：experiment_service / artifact_writer / event_logger / run_state
  diagnostics/                  # health_check / report_exporter / snapshot
  ui/                           # GUI 页签与公共组件
    main_window.py / landing_page.py / fdidm_tab.py / fdidm_plot_widgets.py
    fdidm_adaptive_widgets.py / base_waveform_tab.py / compare_workers.py / ui_utils.py
markdown/                       # 状态/待办文档（REFACTOR_CHECKLIST、TODO）
scripts/                        # check_environment / check_usrp / run_self_test
tests/                          # pytest 行为基线 / 回归测试
.github/workflows/ci.yml        # 纯 Python 核心 CI
```

## 运行

启动后先进入模式选择页，两个入口分别对应独立工作区：

- **仿真**：FDIDM 软波形、OFDM、OTFS、AFDM 与波形对比；先配置参数，再启动仿真并查看结果。
- **硬件验证**：FDIDM 自适应验证与三波形链路测评（OFDM / OTFS / AFDM）；先配置链路并连接 USRP，再启动验证并查看诊断与观测图表。

顶部可返回首页或切换工作区，`Alt+Home` 返回首页，`Ctrl+Q` 退出。离开功能页时自动停止当前仿真、扫描或硬件任务，保留参数与最后结果；硬件连接会释放，返回后需手动连接并启动。页面首次访问时才加载，纯仿真无需连接 USRP。界面继续使用原有 Fusion 浅色控件与 MATLAB 风格图表。

首页使用矢量绘制的仿真与 USRP 图标，配以星地链路、I/Q、频谱和覆盖网络背景，两个入口仍上下排列。硬件配置超出窗口时使用纵向滚动：三波形页各配置页独立滚动，FDIDM 高级参数采用分组两列表单；连接、开始、停止等运行按钮固定可见。长状态和错误保留完整内容，通过滚动区或提示查看，不挤压图表。布局已在 1100×700、1400×900 和不同字体大小下检查。

```powershell
cd waveform_sim
python main.py
```

依赖：Python 3.11+、NumPy、PyQt5、pyqtgraph。硬件链路还需要 GNU Radio / UHD，以及 USRP B210（或 N210 / X310）设备；没有硬件时，纯仿真页面和波形对比页面仍可正常运行。

无显示环境验证：

```powershell
python scripts/check_environment.py   # 环境健康检查
python scripts/run_self_test.py       # FDIDM 自检（出帧 + 实验 artifact）
python -m pytest -q                   # 行为基线 / 回归测试
```

## 平台 AI 对话助手

首页、仿真及硬件验证的所有页面都可点击右下角 **AI** 悬浮球或按 **Ctrl+J** 打开同一个助手。助手由主窗口统一管理，独立于各实验页；可以一边操作平台一边提问，切页后聊天、草稿和缩放仍保留。上方提供“日志分析”“运行结果图解释”“异常原因排查”“优化建议”“操作指导”，点击后会显示对应的用户消息，再给出 AI 回答。输入框支持自由提问，Enter 发送、Shift+Enter 换行。

助手采用较大的统一正文和更宽松的行距，Windows 优先使用微软雅黑 UI，并适配系统显示缩放。标题栏的缩放按钮、Ctrl+滚轮或 Ctrl+加号/减号可整体调整界面（80%–140%），Ctrl+0 恢复 100%。缩放同时作用于已有及后续消息、输入框和按钮；拉伸窗口仅重新排版，不单独缩放某一条回答。

“操作指导”结合当前页真实的按钮名称、可用状态、参数和页面流程，说明下一步点哪里、先满足什么条件以及如何停止。首页或尚未运行实验时也能获得指导；助手只给出建议，不会替用户连接设备、启动测试或修改参数。仿真覆盖 FDIDM、OFDM、OTFS、AFDM 和波形对比，硬件覆盖 FDIDM 自适应验证和三波形链路测评；子视图及硬件页所选波形也纳入来源。

原始日志只作为后台证据，不在界面预览，也不附加到复制或保存的对话中。当前页没有独立日志时，可依据当前图表和保留结果进行分析；也可选择历史 `.log` / `.txt` 文件。“运行结果图解释”读取当前页面实际曲线、坐标轴、统计信息及性能面，区分对数坐标显示、参考曲线与实测结果，不把历史日志当作当前图表数据。切页后的旧回答标明提问时的来源，新的问题使用当前页的快照。分析服务地址、模型和凭据由后台管理。

部署管理员可通过服务端配置提供凭据。Windows 本地部署将凭据以当前 Windows 用户绑定的 DPAPI 加密数据保存在 `%LOCALAPPDATA%\WaveformDemonstrationPlatform\ai-config.dpapi`，不写入项目目录、日志或界面。用于共享部署时，凭据应保留在服务端；本地 DPAPI 配置仅适用于本机部署。管理员或 CI 也可使用 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL` 环境变量覆盖本地配置。

仅在发送问题或点击快捷入口后，后台才会向分析服务提交相关实验上下文。历史日志支持 UTF-8 和 GB18030 编码，上限为 16 MiB；超长日志保留最新 24,000 个字符，并向模型注明截断。对话携带有限的历史记录以支持追问；当前页面记录可能包含多次测试，分析单次实验时可选择对应记录。分析在后台运行，取消或离开页面会忽略未返回的结果；已经发送的请求可能仍由服务端完成。停止测试或返回页面后保留最近日志快照，AI 不会自动修改硬件参数。

## OFDM 星地链路与性能研究

OFDM 页面包含“实时星地链路”和“性能研究”两个视图，支持 NTN-TDLA100 / NTN-TDLC5 的时延与功率轮廓、公共轨道频移预补偿、残余频偏、散射多普勒、分数时延、可配置 FFT / 子载波间隔 / CP，以及导频 LS 或理想单抽头 CSI 参考接收。性能研究按相同发送比特、路径实现和单位噪声对比导频 LS、理想 CSI 与关闭预补偿三种条件，可扫描 Eb/N0、额外残余 CFO、子载波间隔与时延缩放，导出配置和原始统计。

BER 使用真实错误比特数 / 总比特数，不强制单调。零误码仍记录为 0，在对数图上以三角标记给出 95% Wilson 统计上界。Eb/N0 按发送整帧能量 / 载荷比特定义，计入 CP 和导频能量，噪声不随接收衰落逐帧归一化。

该研究模型假设辅助帧定时、未编码 SISO、固定种子路径幅度与初相和恒定逐径多普勒；标准时延/功率轮廓配合简化散射模型，不代表完整 3GPP 一致性信道、完整 NR 协议栈或星地链路预算。固定 FFT 扫描间隔时，带宽与 CP 时长会同时变化；单种子结果是所选信道实现下的结果。波形对比页使用独立的全载波配对基带模型，其曲线与 OFDM 星地研究页采用不同模型，不能直接混作同口径结果。

无界面复现四类扫描：

```powershell
python scripts/study_ofdm_ntn.py --frames 40 --seed 42
```

结果默认保存至 `output/ofdm-ntn-study.json`。模型依据、实现假设和示例分析见 [OFDM 星地性能分析](markdown/OFDM_NTN_ANALYSIS.md)。

## OTFS 星地链路与性能研究

OTFS 页面与 OFDM 对齐为“实时星地链路”和“性能研究”，使用相同的 S / Ka 波段及 AWGN 预设、NTN 时延功率轮廓、公共轨道预补偿、残余 CFO、散射多普勒与分数时延模型。增加时延—多普勒网格尺寸、分辨率和接收 DD 能量展示，可扫描 Eb/N0、残余 CFO、多普勒扩展、子载波间隔、时延缩放与时隙数。

发射端实现矩形脉冲 OTFS 的 ISFFT + 每时隙 CP，接收端 TF 均衡后作 SFFT。每突发增加一个已知训练时隙，以训练 LS 估计信道并保持至整个数据帧；理想 CSI 采用真实主对角项作为参考。两者均为 TF 单抽头 ZF/MMSE，尚未实现完整 DD 均衡或帧内导频跟踪；高动态下的训练老化、ICI 和 ISI 会保留在结果中。

实时与扫描共用数值引擎，三策略共用发送比特、信道和单位噪声。发送整帧能量计入训练与 CP；显示原始 BER/FER/EVM、错误/总比特及 95% Wilson 区间，零误码以统计上界标记绘图，支持不可被后续参数修改的 JSON 研究快照。运行中锁定实验参数，停止或切页保留结果，返回后手动启动。

```powershell
python scripts/study_otfs_ntn.py --frames 40 --seed 42
```

默认输出 `output/otfs-ntn-study.json`。完整假设、接收机限制与解释方式见 [OTFS 星地性能分析](markdown/OTFS_NTN_ANALYSIS.md)。OTFS 的全 DD 载荷加训练时隙，与 OFDM 页的梳状导频预算不同；两页对齐研究流程，不构成同资源波形排名。涉及三波形公平比较时使用下述专门页面，并核对其接收机假设。

## AFDM 星地链路与性能研究

AFDM 页面与 OFDM、OTFS 对齐星地场景、实时链路、性能研究和 JSON 导出流程，增加可编辑 c1/c2、CPP/CP 前缀、数据块数及有效 DAFT 信道矩阵展示。场景支持相同的 NTN 时延/功率轮廓、公共轨道预补偿、残余 CFO、散射多普勒与分数时延。

新研究模型使用标准 DAFT chirp 指数 `exp(-j 2π c n²)`，与旧兼容模型的 `exp(-j 2π c n²/M)` 有明确换算关系；CPP 实际应用 chirp 周期相位修正。已知训练块估计静态信道并构造完整块 LMMSE/ZF 接收机，理想 CSI 单独标注为已知本块时变矩阵的参考。该参考不能代表已经实现实际频偏捕获或完整工程接收机。

性能研究支持 Eb/N0、残余 CFO、多普勒扩展、子载波间隔、时延缩放、c1/c2 与数据块数扫描，保存真实错误/比特、BER/FER/EVM、Wilson 区间及运行时配置。训练和前缀能量计入发送 Eb/N0；取消保留完整点，停止与切页保留最后结果。

```powershell
python scripts/study_afdm_ntn.py --frames 20 --seed 42
```

默认输出 `output/afdm-ntn-study.json`。参数定义、CPP、接收机假设与研究结果见 [AFDM 星地性能分析](markdown/AFDM_NTN_ANALYSIS.md)。三个独立研究页的资源和接收机预算不同；同资源排名应使用下述专门对比页，并核对其模型约定。

## 波形对比的统一口径

波形对比页的实时与扫描共用一套配对引擎。每帧 OFDM / OTFS / AFDM 使用相同载荷比特、物理信道、时域噪声、64×7 全载波资源和含 CP 的发送帧能量；固定种子使重复运行可复现。默认三种波形均采用理想 CSI 完整 LMMSE，另有明确标注的“差异接收机演示”（OFDM 单抽头、OTFS/AFDM 完整 LMMSE）。后者的差异包含接收机预算，不能直接视作波形固有优势。

该页面保持 OFDM / OTFS / AFDM 三波形范围；FDIDM 单独研究软波形参数、特殊参数下的退化关系和自适应过程，不强行并入四波形统一排名。三波形对比的实时与扫描均可修改 AFDM c1/c2，使用标准相位 `exp(-j 2π c n²)`，默认值 `0.00078125 = 0.05/64` 等效保留原波形。参数只影响 AFDM，运行时锁定；结果保留当次 chirp 配置，后续改控件不重标旧曲线。本页仍使用普通 CP，与 AFDM 独立研究页的 CPP/训练结构不同。

界面显示真实错误/比特计数、BER、EVM 与统计区间；零误码在对数图上用统计上界三角标记，原始数据仍为 0。扫描可导出 JSON，包含实际配置、计数、区间和完成/取消/错误状态。运行中锁定参数，实时与扫描互斥；切换页面停止任务并保留完整结果。该模型假设帧定时已知，公共频偏计入理想 CSI，不模拟同步捕获；全载波无导频的通用对比结果不代表完整工程接收机或标准 NTN 一致性测试。

问题复现依据、修复范围和验证方式见 [波形对比页审计](markdown/WAVEFORM_COMPARE_AUDIT.md)。

## 测试

行为基线覆盖统一引擎、波形变换、服务层、硬件抽象与 FDIDM 仿真自适应（含信道块推进、连续多普勒、并发参数修改、时间图平滑、SER-SNR 固定信道对比等回归场景），以及首页导航、页面切换、任务停止和结果保留。CI（`.github/workflows/ci.yml`）安装 NumPy、Matplotlib 和 pytest，以 Agg 无界面后端运行数值与图形导出测试；没有 Qt 依赖时，UI 用例自动跳过。

## 文档

`markdown/` 下保留持续维护的状态文档：
- `REFACTOR_CHECKLIST.md`：重构对照清单（阶段完成状态）
- `TODO.md`：平台待办清单（自适应、对比展示、架构优化）
