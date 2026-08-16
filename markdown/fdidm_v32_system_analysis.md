# FDIDM v32 分析说明：多普勒/时延极限、diag-TF 原因、收发流程与优化

## 1. 当前日志对应的关键参数

日志显示当前主要测试条件为：

- 设备链路：USRP RF / TDL→RF 预渲染后再经过 USRP RF
- 采样率：Fs = 500000 Hz
- FDIDM 网格：M = 16, N = 16
- CP：4 samples
- 调制：QPSK
- 均衡：MMSE
- 编码：Conv1/2, K=7, 171/133 octal, interleaved
- 帧长：756 samples = 1.512 ms
- TX 向量：255840 samples
- 主要测试 TDL：TDL-A, RMS-DS = 1000 ns

核心时间量：

```text
Ts = 1/Fs = 2 us
Δf = Fs/M = 31.25 kHz
Tblock = (M + CP)/Fs = 20/500000 = 40 us
Tdata = N*Tblock = 640 us
Tpreamble_half = L/Fs = 32/500000 = 64 us
Tdiag_obs = (pilot + data)/Fs = 640/500000 = 1.28 ms
TfullH_obs = (M*N*data_frame_len + data_frame_len)/Fs = 164.48 ms
```

当前 v31 日志中，1000 Hz common Doppler 可以大量解码成功；10000 Hz 时同步峰长期低于阈值，说明失败首先发生在同步/CFO 捕获，而不是 FEC 本身。

## 2. 多普勒频偏与扩展极限

### 2.1 common Doppler / CFO 的极限

当前前导是两个相同 half：

```math
s[n] = [a[n], a[n]], \quad L = 32
```

接收端 CFO 估计：

```math
P = \sum_{n=0}^{L-1} r^*[n]r[n+L]
```

```math
\hat f = \frac{F_s}{2\pi L}\angle P
```

因为 `angle()` 只在 `[-π,π)` 内无歧义，所以：

```math
|f| < \frac{F_s}{2L}
```

当前：

```math
f_{unamb} = 500000/(2*32) = 7812.5 Hz
```

别名周期：

```math
f_{period} = F_s/L = 15625 Hz
```

所以 v31 在不做 CFO alias 扫描时，10000 Hz 已经超过 ±7812.5 Hz 无歧义范围，日志中 10000 Hz 长时间无法同步是合理的。v32 新增了 CFO alias scan：先由 repeated-half 得到 alias，再枚举 `alias + k*Fs/L`，用已知前导相关峰选择最佳 CFO。默认扫描 ±50000 Hz，因此 common Doppler 的工程捕获范围从 ±7.8 kHz 扩大到约 ±50 kHz；实际性能还受 SNR、硬件 CFO、TDL fading、同步峰阈值影响。

### 2.2 Doppler spread 的极限

Doppler spread 不是一个统一可完全补偿的 common CFO，它会让信道在 pilot/data 时间内变化。用相位漂移预算：

```math
2\pi f_{spread}T_{obs} \le \theta
```

取严格阈值 `θ=0.35 rad`，宽松阈值 `θ=0.75 rad`。

当前 diag-TF：

```math
T_{obs}=T_{pilot}+T_{data}=1.28 ms
```

```math
f_{spread,strict}=0.35/(2π*1.28ms)=43.5 Hz
```

```math
f_{spread,loose}=0.75/(2π*1.28ms)=93.3 Hz
```

full-H_TF 需要 256 个 one-hot TF probe：

```math
T_{fullH}=256*640us + 640us = 164.48 ms
```

```math
f_{spread,strict}=0.35/(2π*164.48ms)=0.34 Hz
```

```math
f_{spread,loose}=0.75/(2π*164.48ms)=0.73 Hz
```

因此 full-H_TF 只适合静态或极慢变信道；一旦存在 Doppler spread，训练矩阵的不同列不是同一个信道，矩阵会失真。

### 2.3 delay spread / CP 极限

OFDM/TF 对角化要求最大路径时延基本落入 CP：

```math
τ_{max} \le T_{CP}=CP/Fs=4/500000=8 us
```

软件 TDL 内部按照 RMS delay spread 缩放归一化 TDL 表：

```math
τ_i = DS_{rms} \cdot \frac{d_i}{rms(d)}
```

所以：

```math
τ_{max}=DS_{rms}\cdot \frac{d_{max}}{rms(d)}
```

CP 约束给出：

```math
DS_{rms,max}=\frac{T_{CP}}{d_{max}/rms(d)}
```

当前 M=16, CP=4, Fs=500 kHz：

| TDL | dmax/rms(d) | RMS-DS 最大值 | 建议 80% 留量 |
|---|---:|---:|---:|
| TDL-A | 2.842 | 2815 ns | 2252 ns |
| TDL-C | 14.813 | 540 ns | 432 ns |
| TDL-D | 7.334 | 1091 ns | 873 ns |

当前 TDL-A, DS=1000 ns，因此最大路径约 2.84 us，小于 8 us CP，满足 CP 条件。若直接切到 TDL-C 且仍用 DS=1000 ns，则最大路径约 14.8 us，超过 CP，会明显破坏 diag-TF 假设。

## 3. 为什么 diag-TF 在当前系统里最好

理论上，完整 FDIDM/OTFS 类系统在高多普勒时应该利用二维信道结构；但当前工程系统的真实运行条件与理想论文模型不完全一致：

1. 所有 v31/v32 链路都经过真实 USRP RF。真实 RF 包含 USRP 模拟前端、线缆/天线耦合、频偏、采样时钟误差、滤波响应。`tdl_param` 的基只描述可选的软件 TDL 段，不能描述真实 RF 响应。

2. full-H_TF 需要 K=M*N=256 个 probe。当前每个 data/pilot TF frame 为 640 us，训练总时间约 164 ms。只要信道、CFO、SCO 或增益相位在这段时间内变化，256 列就不是同一个 H_TF；之后求 `H = Φ H_TF A` 并做 MMSE/ZF 会放大误差。

3. diag-TF 使用一个密集、恒模 TF pilot，直接估计每个子载波的频响，并在 N 个 pilot OFDM 符号上 LS 平均；N=16 时理论上有约 10log10(16)=12 dB 的估计噪声平均增益。然后再做冲激响应去噪。这与当前短距离 RF/TDL-A 且 CP 覆盖时延的工程信道最匹配。

4. 当前 TDL→RF 是“先软件预渲染，再过真实 RF”。接收端看到的是 `H_total = H_RF * H_TDL_prerendered`，不是纯 TDL，也不是纯论文模型中的单一二维矩阵。所以 diag-TF 更稳并不矛盾；它说明当前实测链路仍主要是 CP-fitting 的近似 LTI/缓慢变化信道。

5. 编码之后 rawBER 可达到 1%～2%，但 Viterbi+CRC 仍能恢复；日志中 1000 Hz 下 rawBER 非零但 FECBER=0，这说明 FEC 正在纠错。EVM 高并不必然导致 CRC 失败。

## 4. 当前收发流程

### 4.1 APP/FEC

发送文本转 UTF-8 bytes，形成：

```text
APP frame = magic(4B) + length(4B) + payload + crc32(4B)
```

当前 payload=`FDIDM OK` 为 8B，因此 uncoded bits：

```math
(4+4+8+4)*8 = 160 bits
```

Conv1/2 编码，K=7，flush 6 bits：

```math
N_{coded}=2*(160+6)=332 bits
```

QPSK 每符号 2 bits，M*N=256 symbols，容量：

```math
N_{cap}=256*2=512 bits
```

剩余 180 bits 用随机 filler 填满，避免每帧完全相同。

### 4.2 FDIDM 变换

设 cross-domain 数据矩阵：

```math
X \in \mathbb{C}^{M \times N}
```

代码中的 FDIDM 发送变换：

```math
X_{TF} = Γ_M(α) X Γ_N(-β)
```

其中：

```math
Γ_N(ε) = \sum_{p=0}^{3} A_p(ε) F_N^p
```

```math
A_p(ε)=\cos((ε-p)π/4)\cos(2(ε-p)π/4)e^{j3(ε-p)π/4}
```

接收反变换：

```math
Y = Γ_M(-α)Y_{TF}Γ_N(β)
```

按列 vec 形式：

```math
x_{TF}=A x,
A = Γ_N(-β) \otimes Γ_M(α)
```

```math
y = Φ y_{TF},
Φ = Γ_N(β) \otimes Γ_M(-α)
```

### 4.3 Heisenberg / Wigner

每个 TF 符号列做 M 点 IFFT 并加 CP：

```math
s_n = CP\{IFFT(X_{TF}[:,n])\}
```

接收端去 CP 后 FFT：

```math
Y_{TF}[:,n] = FFT(r_n[CP:CP+M])
```

### 4.4 帧结构

当前 diag/tdl_param 模式下：

```text
[pre_guard 16]
[sync 64]
[pilot 16*(16+4)=320]
[data 16*(16+4)=320]
[post_guard 36]
```

合计：

```math
16+64+320+320+36 = 756 samples = 1.512 ms
```

### 4.5 同步/CFO/均衡

v32 同步：

1. 用 repeated-half autocorrelation 找 CFO-tolerant 粗同步；
2. 用 known-preamble cross-correlation 细同步；
3. CFO 先由 repeated-half 得 alias：

```math
\hat f_{alias}=\frac{F_s}{2πL}\angle\sum r^*[n]r[n+L]
```

4. 枚举：

```math
\hat f_k = \hat f_{alias}+kF_s/L
```

并用 CFO 校正后的已知前导相关峰选最佳值。

均衡：diag-TF 下先估计：

```math
\hat H[m] = \frac{1}{N}\sum_n \frac{Y_{TF}[m,n]}{X_{pilot,TF}[m,n]}
```

然后对数据：

```math
\hat X_{TF}[m,n] = \frac{\hat H^*[m]}{|\hat H[m]|^2 + σ^2}Y_{TF}[m,n]
```

再做 FDIT 得 cross-domain 符号、硬判决、解交织、Viterbi、CRC。

## 5. v32 代码优化

本次 v32 做了这些优化：

1. 增加参数极限计算 `compute_parameter_limits()`：自动输出 common CFO、Doppler spread、full-H spread、TDL-A/C/D delay spread 的理论预算。
2. 增加 CFO alias scan：解决 `fd=10000 Hz` 超过 repeated-half 无歧义范围时同步失败的问题。
3. 改造同步 metric：保留 sharp cross-corr，同时加入 CFO-insensitive repeated-half autocorr fallback。
4. 改进 residual CFO：由 pilot 第 0/1 个符号扩展为所有相邻 pilot 符号加权平均，并增加残余 CFO 限幅，降低误校正风险。
5. 日志增加 `alias`, `scanScore`, `cfo_unambiguous_hz`, `parameter_limits`，便于后续直接从日志判断是 common CFO 捕获问题还是 spread/CP/equalizer 问题。
6. UI summary/status 增加 CFO alias/scan 信息。

## 6. 建议测试顺序

1. 先跑 RF, fd=0, DS=0，确认 baseline。
2. 跑 TDL-A→RF, DS=1000 ns, fd=0。
3. 固定 spread=0，逐步测 common Doppler：1000, 5000, 10000, 20000 Hz。v32 理论上应能捕获 10000 Hz，若失败看 `scanScore` 和 `sync_metric`。
4. 固定 common Doppler=0 或已补偿，逐步测 Doppler spread：10, 25, 50, 100 Hz。diag-TF 预计 40–100 Hz 是从稳定到边缘的过渡区。
5. 分别测 TDL-A/C/D 的 DS，不要把同一个 1000 ns 套到 TDL-C；TDL-C 建议先从 300–400 ns 开始。

## 7. 本地静态/单元检查

已执行：

```text
python3 -m py_compile fdidm_hardtest.py fdidm_hardware_test_tab.py
```

通过。

在无 GNU Radio 硬件环境下，用合成 repeated preamble 测试 v32 CFO alias scan：

| injected CFO | raw alias 预期 | v32 scan 估计 |
|---:|---:|---:|
| 1000 Hz | 1000 Hz | 1000 Hz |
| 10000 Hz | -5625 Hz | 10000 Hz |
| 20000 Hz | 4375 Hz | 20000 Hz |
| -30000 Hz | 1250 Hz | -30000 Hz |

该测试只验证同步前导 CFO 别名扫描算法，不能替代 B210 实机测试。
