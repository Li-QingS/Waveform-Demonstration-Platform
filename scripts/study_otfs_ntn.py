"""Reproduce the standalone OTFS NTN studies without Qt or radio hardware."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from waveform_sim.simulation.otfs_ntn import OtfsNtnConfig, run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'output' / 'otfs-ntn-study.json')
    args = parser.parse_args()
    if not 1 <= args.frames <= 10000:
        parser.error('--frames must be between 1 and 10000')
    try:
        config = OtfsNtnConfig(seed=args.seed).validated()
    except ValueError as exc:
        parser.error(str(exc))
    experiments = [
        ('ebn0', config, [0., 4., 8., 12., 16., 20.]),
        ('residual_cfo', replace(config, ebn0_db=20.), [0., 300., 1500., 3000., 6000., 15000.]),
        ('doppler', replace(config, ebn0_db=20.), [0., 200., 600., 1200., 2400., 4800.]),
        ('spacing', replace(config, ebn0_db=20.), [15000., 30000., 60000.]),
        ('delay_scale', replace(config, profile='NTN-TDLA100-200', ebn0_db=20.), [1., 5., 10., 25., 50., 100.]),
        ('slots', replace(config, ebn0_db=20.), [4., 8., 12., 24., 48.]),
    ]
    report = {
        'model': 'SISO uncoded rectangular OTFS, ordinary CP per time slot, one known training slot per burst',
        'assumptions': [
            'Assisted frame timing; no acquisition, FEC, RF impairments or link-budget model.',
            'Training-only LS is held across the data slots; the payload is not used to estimate the channel.',
            'Both receivers use TF one-tap ZF/MMSE before SFFT; ideal CSI is the physical main diagonal, not full ICI cancellation.',
            'Transmit burst energy / payload bits defines Eb/N0, including training and CP energy.',
            'Standard NTN delay/power profiles with simplified fixed-amplitude, constant per-path Doppler sum-of-rays fading.',
            'All strategies reuse payload bits, physical paths and unit-noise samples; no forced BER ordering.',
            'Raw error counts and conditional Wilson intervals; one seed is not an ensemble fading-outage estimate.',
            'Fixed FFT spacing sweeps change bandwidth, CP duration and DD resolution together.',
            'This full-grid training budget differs from standalone comb-pilot OFDM and the generic three-waveform comparison.',
        ],
        'frames_per_point': args.frames,
        'experiments': [],
    }
    for axis, cfg, values in experiments:
        points = run_study(cfg, axis, values, args.frames)
        report['experiments'].append({'axis': axis, 'config': asdict(cfg), 'points': points})
        for point in points:
            print(f'{axis}={point["axis_value"]:g}: ' + ', '.join(
                f'{name} BER={stats["ber"]:.5g} ({stats["bit_errors"]}/{stats["total_bits"]})'
                for name, stats in point['strategies'].items()
            ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
