"""Apply the v3 adaptive-history hotfix.

Place this file in the repository root and run:
    python apply_backend_hotfix_v3.py
"""
from pathlib import Path

path = Path("waveform_sim/simulation/fdidm_adaptive.py")
if not path.exists():
    raise SystemExit(f"not found: {path.resolve()}")

text = path.read_text(encoding="utf-8")
original = text

if 'afdm = by_key.get((0.5, 1.0)' not in text:
    needle = (
        '        ofdm = by_key.get((0.0, 0.0), {"ser": float("nan")})\n'
        '        otfs = by_key.get((1.0, 1.0), {"ser": float("nan")})\n'
    )
    if needle not in text:
        raise SystemExit("backend layout changed: reference block not found")
    text = text.replace(
        needle,
        needle + '        afdm = by_key.get((0.5, 1.0), {"ser": float("nan")})\n',
        1,
    )

if '"predicted_ser_afdm"' not in text:
    needle = '            "predicted_ser_otfs": float(otfs.get("ser", float("nan"))),\n'
    if needle not in text:
        raise SystemExit("backend layout changed: result block not found")
    text = text.replace(
        needle,
        needle + '            "predicted_ser_afdm": float(afdm.get("ser", float("nan"))),\n',
        1,
    )

if '"ser_otfs": float(rec["predicted_ser_otfs"])' not in text:
    needle = (
        '                    "ser_ofdm": float(rec["predicted_ser_ofdm"]),\n'
        '                    "gain_db": float(rec["predicted_improvement_db"]),\n'
    )
    replacement = (
        '                    "ser_ofdm": float(rec["predicted_ser_ofdm"]),\n'
        '                    "ser_otfs": float(rec["predicted_ser_otfs"]),\n'
        '                    "ser_afdm": float(rec.get("predicted_ser_afdm", float("nan"))),\n'
        '                    "gain_db": float(rec["predicted_improvement_db"]),\n'
    )
    if needle not in text:
        raise SystemExit("backend layout changed: history block not found")
    text = text.replace(needle, replacement, 1)

if text.count('"predicted_ser_afdm"') < 2:
    needle = (
        '                "predicted_ser_otfs": float(rec.get("predicted_ser_otfs", float("nan"))),\n'
        '                "predicted_improvement_db": float(rec.get("predicted_improvement_db", float("nan"))),\n'
    )
    replacement = (
        '                "predicted_ser_otfs": float(rec.get("predicted_ser_otfs", float("nan"))),\n'
        '                "predicted_ser_afdm": float(rec.get("predicted_ser_afdm", float("nan"))),\n'
        '                "predicted_improvement_db": float(rec.get("predicted_improvement_db", float("nan"))),\n'
    )
    if needle not in text:
        raise SystemExit("backend layout changed: status block not found")
    text = text.replace(needle, replacement, 1)

if text == original:
    print("backend already contains v3 history fields:", path)
else:
    backup = path.with_suffix(path.suffix + ".v3.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print("patched:", path)
    print("backup :", backup)
