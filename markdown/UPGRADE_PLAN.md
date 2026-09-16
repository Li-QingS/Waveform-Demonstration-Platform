# 升级计划:从展示原型到可信实验平台

> 创建日期:2026-09-05(与 ZCode 讨论定稿)
> 性质:下一阶段施工总纲,覆盖波形对比、证据分级、自适应状态机、硬件稳定四个方向。
> 一句话目标:让平台能回答——**在相同带宽/功率/净速率/信道/接收复杂度约束下,FDIDM 比谁好、好多少、为什么好,并且仿真结论有统计依据、硬件链路跑得稳。**

---

## 问题 1:统一波形对比考卷(根问题)

**现状**:两套对比代码口径互相矛盾——理论对比(`shared_waveform_benchmark.py`)有 FDIDM 但只算预测值;MC 扫描(`compare_scan_backend.py`)算真实比特却**没有 FDIDM**、三种波形接收机复杂度不对齐(OTFS 整帧 LMMSE vs OFDM 单抽头)、随机种子里编入了波形序号导致各波形信道样本不同、SNR 定义有两套。

**已定决策**:
- 两种实验都做:先"系统级"(各自标准接收机 + 报告复杂度),后"等预算"(相同检测器复杂度);
- 开销口径:固定帧资源,净速率作为独立维度随 BER 一起报告;
- 老对比代码**推倒重写**,不做 legacy 兼容。

**任务**:
- [ ] 新建 `waveform_sim/simulation/waveform_benchmark.py`:`ExperimentContext`(帧结构/信道参数/SNR 定义/种子规则,全平台唯一)+ 4 个 `WaveformAdapter`(变换、资源映射、标准接收机、复杂度报告)+ 统一执行器
- [ ] 抢救三块资产再删旧文件:`equivalent_channel`(四波形 H_eff 推导)、`optimize_fdidm_over_ensemble`(α/β 搜索内核,自适应闭环依赖它)、interference-aware LMMSE(失配协方差写法正确)
- [ ] 两条铁律:**种子 = f(实验ID, 帧号),不含波形号**(四波形逐比特同信道同噪声);**SNR 定义全平台只有一份**
- [ ] 删除 `shared_waveform_benchmark.py`、`compare_scan_backend.py`;`fdidm_adaptive.py` / `ui/fdidm_tab.py` / `ui/compare_workers.py` 改 import
- [ ] 输出统一 JSON:`level / ber / ser / net_rate / complexity(每数据符号复乘数) / seed / channel_id`
- [ ] Phase 1(系统级):QPSK + 完美 CSI 起步,替换波形对比页数据源
- [ ] Phase 2(等预算):复杂度预算分配器 + 稀疏接收机等预算曲线

**验收**:
- 同种子下四波形比特流与信道快照逐位相同(新增 pytest);
- OFDM 在 AWGN 下 BER 贴理论曲线(整链 sanity);
- 现有 88 个回归用例(含自适应)不回归。

---

## 问题 2:结果分级(证据等级)

**现状**:理论算的、仿真跑的、硬件测的数字混在一起展示,不标等级;链路异常时照常算"优势百分比"。历史教训见 `specs/fdidm-hardware-stability/spec.md`:预测 SER 有增益,应用后实测无改善。

**已定决策**:
- `simulated` / `measured` 可生成"好 X dB"类结论,`predicted` 永远只作"预期";
- `invalid` 数据不进结论,界面灰标 + 显示原因(overflow/失步/样本不足);
- 统计门槛**照标准,不自创**。

**任务**:
- [ ] 统一结果 schema 增加必填字段 `level`(四级:predicted / simulated / measured / invalid)
- [ ] 仿真/实测停止条件:错误数 ≥ 100 或比特数达上限,先到为准(MATLAB 官方建议:至少 100 个错误)
- [ ] 每条 BER 结果必记错误数 k 与总比特数 n;置信区间 `p̂ ± 1.96·√(p̂(1−p̂)/n)`
- [ ] 可信度自动降级:k ≥ 100 可信;10 ≤ k < 100 标"参考值"(灰);k < 10 归 invalid
- [ ] UI:结果按等级打徽标;invalid 灰色/空心标记并给出原因;链路异常(overflow/失步/积压)时实验整体判 invalid

**参考标准**:
- MathWorks《Bit Error Rate Analysis Techniques》:"simulating enough data to produce at least 100 errors provides accurate error rate results"
- NASA JPL IPN Progress Report 42-201:±100γ% 相对精度(95% 置信)需 4/γ² 个错误 → ±10% 需 400 个错
- 硬件侧同源:Keysight BERT 置信度测试即同一泊松/二项统计

---

## 问题 3:自适应闭环改确定性状态机(先仿真侧)

**现状**:仿真侧自适应待处理数据只有单槽位(`fdidm_adaptive.py:126`),搜索慢了新帧覆盖旧帧,评估与推荐的时序依赖线程调度——同样实验跑两遍轨迹可能不同。

**已定决策**:
- 积压策略:**只保留最新快照;搜索结果按"统计上下文键"判过期,不按"具体信道实现"判过期**(`_adaptive_context_key` 机制已存在,`fdidm_adaptive.py:188`,16 项统计参数;commit 4e658fb 已解决"全丢→饿死→永不切换");键不变的搜索结果正常复用,由 VALIDATING 环节兜底;
- 范围:先改仿真侧;硬件侧已有 spec(F3~F5 validating/回滚),等长稳基线解决后对齐同一套词汇。

**任务**:
- [ ] 显式状态机:`COLLECTING → SEARCHING → RECOMMENDED → APPLYING → SETTLING → VALIDATING → ACCEPTED / ROLLBACK → COOLDOWN`,仿真/硬件两侧同名同义
- [ ] 所有状态转移由**帧序号**驱动,禁止依赖线程相对速度
- [ ] 搜索完成时比对上下文键:不变 → RECOMMENDED;变 → 作废重搜
- [ ] VALIDATING:切换瞬态丢弃后,用真实帧统计验证,恶化/失步/超时 → 回滚

**验收**:固定种子连跑两遍,自适应决策轨迹(每次评估的帧号、推荐值、切换点)**逐位一致**。

---

## 问题 4/5:硬件方向(小步慢拆 + 仅 FDIDM 链)

**现状**:`fdidm_hardtest.py` 4591 行单文件(UHD 流图/同步/CFO/信道估计/均衡/FEC/自适应/指标/线程全在内);`transport.py`、`iq_replay.py`、`rf_safety.py` 已写但**零引用**;RX 曾出现 overflow/积压,其波动远大于 α/β 优化收益。

**已定决策**:
- IQ 回放先行(一石三鸟:校准台阶 + 硬件代码离线测试手段 + 拆分安全网);
- 小步慢拆:先只拆 ReceiverPipeline + Transport 两块,其余边长稳边拆;
- 硬件对比链**仅 FDIDM**;实测 A/B 对比推迟,"FDIDM vs OFDM 谁好"的结论现阶段由仿真证据链支撑,硬件负责验证 FDIDM 链路本身稳定。

**任务**:
- [ ] 把 `iq_replay.py` 接进 Transport:硬件录 IQ → 软件接收机离线跑,逐级定位(射频时钟 / 同步估计 / FDIDM 变换 / 均衡 / 自适应决策)
- [ ] 用录制 IQ 建立硬件代码的锚定测试(拆分前后行为不变的依据)
- [ ] 拆出 `ReceiverPipeline`(同步/估计/均衡/译码)与 `Transport`(仿真 / IQ回放 / UHD 三实现)
- [ ] `rf_safety.py` 接入真实执行入口强制生效
- [ ] 长稳阶梯(需 USRP 在手):
  - 阶段 0:固定 FDIDM(默认 α/β)长稳——验收:连续 ≥2h、overflow=0、CRC 通过帧 ≥99%、失步 <0.1%
  - 阶段 1:启用自动搜索/应用/回滚,并验证收益在 measured 层级成立
- [ ] 校准台阶全链:数学单测 → MC 仿真 → IQ 回放 → GNU Radio 软环回 → 同轴线缆/衰减器 → 空口静态 → 动态信道+自适应

---

## 施工顺序(依赖关系)

```text
1. 问题1 Phase 1 统一对比模块      ← 纯软件,立即开工,是 2 的前置
2. 问题2 结果分级 schema           ← 依赖 1 的输出结构
3. 问题3 仿真侧状态机              ← 可与 2 并行
4. IQ 回放接线 + 锚定测试          ← 有硬件在手即可,可与 2/3 并行
5. 硬件小拆(ReceiverPipeline/Transport)← 依赖 4 的安全网
6. 长稳阶梯 阶段0→阶段1            ← 依赖 5
7. 问题1 Phase 2 等预算对比        ← 可与 4~6 并行
```

## 明确不做的事

- 不新增页面、不新增算法选项,先收敛证据链;
- 不做 OTFS/AFDM 硬件链;
- 不做硬件 A/B 对比(待 FDIDM 链长稳且确有需要再议);
- 不自创统计门槛,全部照标准;
- 不引入第二台 USRP / 并行射频链路(沿用 stability spec N1)。

---

## 方向更新(2026-09-07)

应用户要求,"硬件展示不能体现软波形优势"需提前解决:本计划"不做硬件 A/B 对比"的决策**部分解除**——已完成「FDIDM 硬件优势观测升级」(spec/plan/task/checklist 见 `specs/fdidm-hardware-advantage-observability/`)。

- 机制为**评审手动开关自适应 + 页面自动配对统计**(未做自动定时 A/B 交替,"不做自动 A/B"仍然成立);
- 证据分级(错误数 ≥100 可信 / 10–99 参考 / <10 无效)与本计划问题 2 同源;
- 硬件实测对比结论措辞受"统计可比而非逐样本可比"约束。
