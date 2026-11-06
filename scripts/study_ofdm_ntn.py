"""Reproduce the OFDM NTN parameter study without Qt or radio hardware."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from waveform_sim.simulation.ofdm_ntn import OfdmNtnConfig, run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'output' / 'ofdm-ntn-study.json')
    args = parser.parse_args()
    if not 1 <= args.frames <= 10000:
        parser.error('--frames must be between 1 and 10000')
    config = OfdmNtnConfig(seed=args.seed).validated()
    experiments = [
        ('ebn0', config, [0., 4., 8., 12., 16., 20.]),
        ('residual_cfo', replace(config, ebn0_db=20.), [0., 300., 1500., 3000., 6000., 15000.]),
        ('spacing', replace(config, ebn0_db=20.), [15000., 30000., 60000.]),
        ('delay_scale', replace(config, profile='NTN-TDLA100-200', ebn0_db=20.), [1., 5., 10., 25., 50., 100.]),
    ]
    report = {
        'model': 'SISO uncoded CP-OFDM, standard NTN delay/power profiles with simplified sum-of-rays fading',
        'assumptions': [
            'Assisted, aligned frame timing; no acquisition/outage or link-budget model.',
            'Transmit burst energy / payload bits defines Eb/N0, including CP and pilot energy.',
            'Raw bit counts; Wilson 95% intervals are conditional counting estimates, not fading-outage confidence bounds.',
            'One seed fixes path amplitudes and initial phases; constant per-path Doppler rotates phases continuously.',
            'All variants reuse bits, physical paths and unit-noise streams; no forced monotonic curves.',
            'Fixed FFT in the spacing study changes bandwidth and CP duration; fixed frames, not fixed elapsed time.',
        ],
        'frames_per_point': args.frames, 'experiments': [],
    }
    for axis, cfg, values in experiments:
        points = run_study(cfg, axis, values, args.frames)
        report['experiments'].append({'axis': axis, 'config': asdict(cfg), 'points': points})
        for point in points:
            stats = point['strategies']
            print(f'{axis}={point["axis_value"]:g}: ' + ', '.join(
                f'{name} BER={s["ber"]:.5g} ({s["bit_errors"]}/{s["total_bits"]})'
                for name, s in stats.items()
            ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
