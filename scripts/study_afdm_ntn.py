"""Reproduce AFDM satellite-terrestrial studies without Qt or radio hardware."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from waveform_sim.simulation.afdm_ntn import AfdmNtnConfig, config_summary, run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'output' / 'afdm-ntn-study.json')
    args = parser.parse_args()
    if not 1 <= args.frames <= 10000:
        parser.error('--frames must be between 1 and 10000')
    try:
        config = AfdmNtnConfig(seed=args.seed).validated()
    except ValueError as exc:
        parser.error(str(exc))
    experiments = [
        ('ebn0', config, [0., 4., 8., 12., 16., 20.]),
        ('residual_cfo', replace(config, ebn0_db=20.), [0., 300., 1500., 6000., 15000.]),
        ('doppler', replace(config, ebn0_db=20.), [0., 200., 1200., 4800.]),
        ('spacing', replace(config, ebn0_db=20.), [15000., 30000., 60000.]),
        ('delay_scale', replace(config, profile='NTN-TDLA100-200', ebn0_db=20.), [1., 10., 25., 50.]),
        ('c1', replace(config, ebn0_db=20.), [0., 1. / (2 * config.fft_len), 3. / (2 * config.fft_len), .02]),
        ('c2', replace(config, ebn0_db=20.), [0., .0001, .001, .01]),
        ('blocks', replace(config, ebn0_db=20.), [1., 4., 8., 16.]),
    ]
    report = {
        'model': 'SISO uncoded AFDM, standard DAFT chirp coefficients and configurable CPP/CP',
        'assumptions': [
            'Aligned frame timing and configured orbital pre-compensation, not blind acquisition.',
            'DAFT chirps use exp(-j*2*pi*c*n^2), unlike legacy coefficients divided by the FFT length.',
            'Known training provides the static channel estimate; payload symbols only score decisions.',
            'Full within-block linear equalization; ideal CSI is a labelled reference and does not remove other-block ISI.',
            'Transmit burst energy / payload bits defines Eb/N0, including training and prefix energy.',
            'Standard NTN delay/power profiles with simplified fixed-amplitude, constant-Doppler ray fading.',
            'Common bits, physical paths and unit-noise streams across strategies; no imposed ranking or monotonicity.',
            'Raw zero BER and conditional Wilson 95% counting intervals; one seed is not a fading-outage estimate.',
            'Changing chirps changes the waveform and its actual prefix energy; changing spacing also changes bandwidth and prefix duration.',
            'Standalone OFDM/OTFS and the generic comparison use different resource/receiver budgets.',
        ],
        'frames_per_point': args.frames, 'experiments': [],
    }
    for axis, cfg, values in experiments:
        points = run_study(cfg, axis, values, args.frames)
        report['experiments'].append({'axis': axis, 'config': asdict(cfg), 'summary': config_summary(cfg), 'points': points})
        for point in points:
            print(f'{axis}={point["axis_value"]:g}: ' + ', '.join(
                f'{name} BER={s["ber"]:.5g} ({s["bit_errors"]}/{s["total_bits"]})'
                for name, s in point['strategies'].items()
            ), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
