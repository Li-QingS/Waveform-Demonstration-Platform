"""Behavioral contract for static operator guidance."""

import json

from waveform_sim.service.platform_guidance import get_page_guide


PAGE_IDS = (
    "home",
    "simulation.fdidm",
    "simulation.ofdm",
    "simulation.otfs",
    "simulation.afdm",
    "simulation.compare",
    "hardware.fdidm",
    "hardware.link",
)


def _workflow(guide, name):
    return next(item for item in guide["workflows"] if item["name"] == name)


def test_ofdm_guidance_gives_real_time_and_research_operator_sequences():
    guide = get_page_guide("simulation.ofdm")

    assert _workflow(guide, "实时星地链路")["steps"] == [
        "在“星地场景”和“波形与接收”中确认参数。",
        "点击“开始仿真”。",
        "在“实时星地链路”查看链路指标、频谱和星座。",
        "点击“停止”结束当前链路。",
    ]
    assert _workflow(guide, "性能研究")["steps"] == [
        "切换到“性能研究”，设置“研究变量”“扫描取值”和“每点帧数”。",
        "点击“开始性能扫描”。",
        "完成后查看 BER、EVM、统计上界和结果表。",
        "点击“导出研究结果 (JSON)”保存研究快照。",
    ]


def test_fdidm_guidance_uses_automatic_refresh_and_only_applies_visible_ready_result():
    guide = get_page_guide("simulation.fdidm")

    assert _workflow(guide, "自动理论刷新")["steps"] == [
        "修改页面参数，或点击“重生成信道”。",
        "等待自动理论搜索及 α-SER、SER-SNR 曲线刷新完成。",
        "仅当“应用最优”可用且结果符合当前实验条件时，点击“应用最优”。",
    ]
    assert all(
        "理论搜索最优 α/β" not in step
        for workflow in guide["workflows"]
        for step in workflow["steps"]
    )


def test_compare_guidance_requires_paired_metadata_for_fair_results():
    guide = get_page_guide("simulation.compare")

    assert _workflow(guide, "实时对比")["steps"] == [
        "在“实时对比”选择左右波形，并确认“接收机口径”和“随机种子”。",
        "点击“开始对比”。",
        "记录结果中显示的 Eb/N0、CFO、多普勒、seed、接收机口径和 AFDM c1/c2。",
        "点击“停止”后再修改参数或开始下一组比较。",
    ]
    assert any("相同比特、信道、噪声及发送能量" in caveat for caveat in guide["caveats"])


def test_hardware_link_guidance_follows_explicit_connect_start_stop_and_release_flow():
    guide = get_page_guide("hardware.link")

    assert _workflow(guide, "三波形链路测评")["steps"] == [
        "确认 USRP、天线、外接衰减与所选“设备型号”“中心频率”一致。",
        "点击“连接 / 配置 USRP”，确认状态为已配置。",
        "需要修改已连接参数时，点击“应用当前参数”。",
        "点击“开始测试”，观察“文本对照”“测试日志”和诊断图。",
        "点击“停止测试”结束采集。",
        "点击“释放 USRP”结束本次设备占用。",
    ]
    assert any("增益" in caveat and "确认" in caveat for caveat in guide["caveats"])


def test_fdidm_hardware_guidance_defers_iq_import_to_visible_context_and_exports_log():
    guide = get_page_guide("hardware.fdidm")

    assert _workflow(guide, "FDIDM 自适应验证")["steps"] == [
        "确认 USRP、天线、外接衰减和必要的硬件安全条件。",
        "点击“连接/配置”，确认设备已就绪。",
        "点击“开始测试”，在“优势观测”或“链路诊断”查看结果。",
        "点击“停止测试”结束当前测试。",
        "需要保存证据时点击“导出日志”。",
    ]
    assert any("IQ" in caveat and "实际显示" in caveat for caveat in guide["caveats"])
    assert _workflow(guide, "高级参数调整")["steps"] == [
        "勾选“显示高级参数”。",
        "使用“推荐 0.5/1”“OFDM 0/0”或“OTFS 1/1”选择明确的 α/β 对照。",
        "仅在理解当前链路与观测结果后点击“应用参数”或“立即评估”。",
    ]


def test_all_known_pages_return_distinct_json_serializable_guides():
    guides = [get_page_guide(page_id) for page_id in PAGE_IDS]

    assert all(set(guide) == {"title", "overview", "workflows", "caveats"} for guide in guides)
    assert all(guide["workflows"] and guide["caveats"] for guide in guides)
    assert len({json.dumps(guide, ensure_ascii=False, sort_keys=True) for guide in guides}) == len(PAGE_IDS)


def test_unknown_page_returns_safe_navigation_guidance_without_claiming_controls():
    guide = get_page_guide("simulation.unknown")

    assert guide["title"] == "页面操作指引"
    assert guide["workflows"] == [{
        "name": "安全导航",
        "steps": [
            "返回首页或切换到已识别的功能页。",
            "根据当前页面实际显示的控件状态和提示继续操作。",
        ],
    }]
    assert any("不执行" in caveat for caveat in guide["caveats"])
