"""Static operator guidance for platform pages.

This module intentionally has no Qt dependency.  A UI/context adapter can
attach current enabled-state information without changing the durable page
workflow guidance returned here.
"""

from copy import deepcopy


_SHARED_NAVIGATION_CAVEAT = (
    "切换页面、工作区或返回首页会停止当前实验但保留参数和最后结果；返回后不会自动启动。"
)
_STATE_CAVEAT = "以当前界面实际显示的控件状态、状态栏和错误提示为准；本指引不操作任何控件。"


def _guide(title, overview, workflows, caveats):
    return {
        "title": title,
        "overview": overview,
        "workflows": [
            {"name": name, "steps": list(steps)} for name, steps in workflows
        ],
        "caveats": list(caveats),
    }


_GUIDES = {
    "home": _guide(
        "平台首页",
        "从首页进入仿真或硬件验证工作区；页面只在首次访问时加载。",
        (
            ("进入仿真", (
                "选择“仿真”。",
                "在 FDIDM 软波形、OFDM、OTFS、AFDM 或“波形对比”页中选择任务。",
            )),
            ("进入硬件验证", (
                "选择“硬件验证”。",
                "在“FDIDM 自适应验证”或“三波形链路测评”页中选择任务。",
            )),
        ),
        (_SHARED_NAVIGATION_CAVEAT, _STATE_CAVEAT),
    ),
    "simulation.fdidm": _guide(
        "FDIDM 软波形",
        "用于 FDIDM 实时仿真、理论 α/β 搜索和自适应过程观察。",
        (
            ("实时仿真", (
                "确认页面参数与时变信道设置。",
                "点击“开始仿真”。",
                "在“仿真图”查看实时链路状态，在“自适应过程”查看 α/β 与 SER-SNR 结果。",
                "点击“停止”结束当前仿真。",
            )),
            ("自动理论刷新", (
                "修改页面参数，或点击“重生成信道”。",
                "等待自动理论搜索及 α-SER、SER-SNR 曲线刷新完成。",
                "仅当“应用最优”可用且结果符合当前实验条件时，点击“应用最优”。",
            )),
        ),
        (
            "理论 GMI/SER 结果与硬件 ABBA 实测是不同证据；不要把二者当作同一测量。",
            _SHARED_NAVIGATION_CAVEAT,
            _STATE_CAVEAT,
        ),
    ),
    "simulation.ofdm": _guide(
        "OFDM 星地链路",
        "用于 OFDM 实时星地链路演示和可复现性能研究。",
        (
            ("实时星地链路", (
                "在“星地场景”和“波形与接收”中确认参数。",
                "点击“开始仿真”。",
                "在“实时星地链路”查看链路指标、频谱和星座。",
                "点击“停止”结束当前链路。",
            )),
            ("性能研究", (
                "切换到“性能研究”，设置“研究变量”“扫描取值”和“每点帧数”。",
                "点击“开始性能扫描”。",
                "完成后查看 BER、EVM、统计上界和结果表。",
                "点击“导出研究结果 (JSON)”保存研究快照。",
            )),
        ),
        (
            "星地研究页和波形对比页使用不同模型与资源；没有匹配的预算和假设时，不能跨页排名。",
            _SHARED_NAVIGATION_CAVEAT,
            _STATE_CAVEAT,
        ),
    ),
    "simulation.otfs": _guide(
        "OTFS 星地链路",
        "用于 OTFS 实时星地链路演示和延迟-多普勒性能研究。",
        (
            ("实时星地链路", (
                "在“星地场景”和“波形与接收”中确认参数。",
                "点击“开始仿真”。",
                "在“实时星地链路”观察 DD 能量、链路指标和星座。",
                "点击“停止”结束当前链路。",
            )),
            ("性能研究", (
                "切换到“性能研究”，设置研究变量、扫描取值和每点帧数。",
                "点击“开始性能扫描”，需要中断时点击“停止扫描”。",
                "完成后点击“导出研究结果 (JSON)”保存配置和原始统计。",
            )),
        ),
        (
            "该研究接收机的训练和均衡假设应随导出的配置一起解释；不能直接与其他页面的不同预算曲线比较。",
            _SHARED_NAVIGATION_CAVEAT,
            _STATE_CAVEAT,
        ),
    ),
    "simulation.afdm": _guide(
        "AFDM 星地链路",
        "用于 AFDM 实时星地链路演示、chirp 参数建议和性能研究。",
        (
            ("实时星地链路", (
                "在“星地场景”和“波形与接收”中确认参数。",
                "需要时点击“应用场景建议 chirp”。",
                "点击“开始仿真”，在“实时星地链路”观察结果。",
                "点击“停止”结束当前链路。",
            )),
            ("性能研究", (
                "切换到“性能研究”，设置研究变量、扫描取值和每点帧数。",
                "点击“开始性能扫描”，完成后查看 BER、EVM 和结果表。",
                "点击“导出研究结果 (JSON)”保存研究快照。",
            )),
        ),
        (
            "chirp 建议是场景参数的起点，仍需连同信道、接收机和资源预算解释结果。",
            _SHARED_NAVIGATION_CAVEAT,
            _STATE_CAVEAT,
        ),
    ),
    "simulation.compare": _guide(
        "波形对比",
        "用于以相同帧源进行的配对实时对比和 CFO/多普勒自动扫描。",
        (
            ("实时对比", (
                "在“实时对比”选择左右波形，并确认“接收机口径”和“随机种子”。",
                "点击“开始对比”。",
                "记录结果中显示的 Eb/N0、CFO、多普勒、seed、接收机口径和 AFDM c1/c2。",
                "点击“停止”后再修改参数或开始下一组比较。",
            )),
            ("自动扫描", (
                "切换到“自动扫描”，确认每点统计帧数、接收机口径和随机种子。",
                "点击“开始自动扫描”。",
                "只将已完成的配对点用于比较；需要中断时点击“停止扫描”。",
                "扫描完成后点击“导出结果”。",
            )),
        ),
        (
            "配对比较使用相同比特、信道、噪声及发送能量；公平性还取决于保留显示的接收机口径和配置元数据。",
            "零误码标记表示统计上界，不能改写为非零实测 BER。",
            _SHARED_NAVIGATION_CAVEAT,
            _STATE_CAVEAT,
        ),
    ),
    "hardware.fdidm": _guide(
        "FDIDM 自适应验证",
        "用于 USRP 上的 FDIDM 自适应链路验证、诊断和日志证据导出。",
        (
            ("FDIDM 自适应验证", (
                "确认 USRP、天线、外接衰减和必要的硬件安全条件。",
                "点击“连接/配置”，确认设备已就绪。",
                "点击“开始测试”，在“优势观测”或“链路诊断”查看结果。",
                "点击“停止测试”结束当前测试。",
                "需要保存证据时点击“导出日志”。",
            )),
            ("高级参数调整", (
                "勾选“显示高级参数”。",
                "使用“推荐 0.5/1”“OFDM 0/0”或“OTFS 1/1”选择明确的 α/β 对照。",
                "仅在理解当前链路与观测结果后点击“应用参数”或“立即评估”。",
            )),
        ),
        (
            "不要盲目调整 RF 增益；先确认天线、外接衰减、设备状态和现场安全条件。",
            "仅当上下文实际显示 IQ 导入或回放入口且已有对应 IQ 文件时才使用它；本静态指引不会假定该控件存在。",
            _SHARED_NAVIGATION_CAVEAT,
            _STATE_CAVEAT,
        ),
    ),
    "hardware.link": _guide(
        "三波形链路测评",
        "用于 OFDM、OTFS 和 AFDM 的 USRP 链路配置、测试和现场诊断。",
        (
            ("三波形链路测评", (
                "确认 USRP、天线、外接衰减与所选“设备型号”“中心频率”一致。",
                "点击“连接 / 配置 USRP”，确认状态为已配置。",
                "需要修改已连接参数时，点击“应用当前参数”。",
                "点击“开始测试”，观察“文本对照”“测试日志”和诊断图。",
                "点击“停止测试”结束采集。",
                "点击“释放 USRP”结束本次设备占用。",
            )),
            ("文本负载", (
                "使用“加载文本文件”或编辑发送文本。",
                "测试后在“文本对照”核对发送与接收内容，再查看“测试日志”。",
            )),
        ),
        (
            "不要盲目调整 RF 增益；先确认天线、外接衰减、设备状态和现场安全条件。",
            "当前页面未显示 IQ 导入或结果导出控件；仅在上下文显示相应入口且已有文件时，按该入口提示操作。",
            _SHARED_NAVIGATION_CAVEAT,
            _STATE_CAVEAT,
        ),
    ),
}


_UNKNOWN_GUIDE = _guide(
    "页面操作指引",
    "该页面没有静态专用指引；请根据实际可见控件和状态提示导航。",
    (("安全导航", (
        "返回首页或切换到已识别的功能页。",
        "根据当前页面实际显示的控件状态和提示继续操作。",
    )),),
    (
        "本指引不执行、启用或禁用任何控件，也不启动仿真或硬件操作。",
        _SHARED_NAVIGATION_CAVEAT,
        _STATE_CAVEAT,
    ),
)


def get_page_guide(page_id: str) -> dict:
    """Return JSON-serializable, static operator guidance for *page_id*.

    Unknown or malformed identifiers receive a conservative navigation guide.
    A copy is returned so callers may enrich it with live UI context safely.
    """
    if not isinstance(page_id, str):
        return deepcopy(_UNKNOWN_GUIDE)
    return deepcopy(_GUIDES.get(page_id, _UNKNOWN_GUIDE))


__all__ = ["get_page_guide"]
