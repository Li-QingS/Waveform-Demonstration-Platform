from pathlib import Path

path = Path("waveform_sim/simulation/fdidm_adaptive.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    '        ofdm = by_key.get((0.0, 0.0), {"ser": float("nan")})\n'
    '        otfs = by_key.get((1.0, 1.0), {"ser": float("nan")})\n',
    '        ofdm = by_key.get((0.0, 0.0), {"ser": float("nan")})\n'
    '        otfs = by_key.get((1.0, 1.0), {"ser": float("nan")})\n'
    '        afdm = by_key.get((0.5, 1.0), {"ser": float("nan")})\n'
)

text = text.replace(
    '            "predicted_ser_otfs": float(otfs.get("ser", float("nan"))),\n',
    '            "predicted_ser_otfs": float(otfs.get("ser", float("nan"))),\n'
    '            "predicted_ser_afdm": float(afdm.get("ser", float("nan"))),\n'
)

text = text.replace(
    '                    "ser_ofdm": float(rec["predicted_ser_ofdm"]),\n'
    '                    "gain_db": float(rec["predicted_improvement_db"]),\n',
    '                    "ser_ofdm": float(rec["predicted_ser_ofdm"]),\n'
    '                    "ser_otfs": float(rec["predicted_ser_otfs"]),\n'
    '                    "ser_afdm": float(rec.get("predicted_ser_afdm", float("nan"))),\n'
    '                    "gain_db": float(rec["predicted_improvement_db"]),\n'
)

text = text.replace(
    '                "predicted_ser_otfs": float(rec.get("predicted_ser_otfs", float("nan"))),\n'
    '                "predicted_improvement_db": float(rec.get("predicted_improvement_db", float("nan"))),\n',
    '                "predicted_ser_otfs": float(rec.get("predicted_ser_otfs", float("nan"))),\n'
    '                "predicted_ser_afdm": float(rec.get("predicted_ser_afdm", float("nan"))),\n'
    '                "predicted_improvement_db": float(rec.get("predicted_improvement_db", float("nan"))),\n'
)

path.write_text(text, encoding="utf-8")
print("patched", path)
