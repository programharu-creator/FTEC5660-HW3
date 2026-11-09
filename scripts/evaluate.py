'''Run reproducible public benchmarks through the official command line entry.'''

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Evaluate the Fisher agent on the public companies')
    parser.add_argument('--mode', choices=['baseline', 'reflection'], default='reflection')
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    reports = ROOT / 'reports'
    private = ROOT / '.evaluation'
    reports.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    entries = []
    for run in range(1, args.runs + 1):
        label = f'{args.mode}_{run}'
        trace_path = private / f'{label}.json'
        env = dict(os.environ, HW3_REFLECTION='1' if args.mode == 'reflection' else '0',
                   HW3_TRACE_FILE=str(trace_path), PYTHONUTF8='1')
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, '-X', 'utf8', 'hw3.py', '--data-folder', 'public_test'],
            cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', timeout=1700)
        seconds = time.perf_counter() - started
        (private / f'{label}.log').write_text(process.stdout + process.stderr, encoding='utf-8')
        if process.returncode:
            raise RuntimeError(f'Official run failed with exit code {process.returncode}; see the local log')
        print(process.stdout.strip(), flush=True)
        with (ROOT / 'results.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        (reports / f'results_{label}.csv').write_bytes((ROOT / 'results.csv').read_bytes())
        truth = json.loads((ROOT / 'public_test/ground_truth.json').read_text(encoding='utf-8'))
        traces = json.loads(trace_path.read_text(encoding='utf-8'))
        counts = {point: {'PASS': [0, 0], 'FAIL': [0, 0]} for point in [4, 7, 11]}
        disagreements = []
        for row in rows:
            verdicts = {int(token.split(':')[0]): token.split(':')[1] for token in row['points'].split()}
            for point, value in verdicts.items():
                expected = truth[row['company']]['points'][str(point)]
                counts[point][expected][0] += value == expected
                counts[point][expected][1] += 1
                if value != expected:
                    disagreements.append({'company': row['company'], 'point': point,
                                          'expected': expected, 'predicted': value})
        passing = [sum(counts[p]['PASS'][i] for p in counts) for i in [0, 1]]
        failing = [sum(counts[p]['FAIL'][i] for p in counts) for i in [0, 1]]
        calls = [call for trace in traces.values() for call in trace['calls']]
        entry = {
            'label': label, 'mode': args.mode, 'run': run, 'companies': len(rows),
            'correct': sum(int(row['points_correct'].split('/')[0]) for row in rows),
            'total': len(rows) * 3,
            'balanced_accuracy': (passing[0] / passing[1] + failing[0] / failing[1]) / 2,
            'pass_recall': passing[0] / passing[1], 'fail_recall': failing[0] / failing[1],
            'seconds': round(seconds, 3), 'model_calls': len(calls),
            'input_tokens': sum(call.get('usage', {}).get('input_tokens', 0) for call in calls),
            'output_tokens': sum(call.get('usage', {}).get('output_tokens', 0) for call in calls),
            'interviews': sum(int(row['interviews_used']) for row in rows),
            'max_interviews_per_company': max(int(row['interviews_used']) for row in rows),
            'failed_companies': sum(trace['status'] == 'failed' for trace in traces.values()),
            'degraded_companies': sum(trace['status'] == 'degraded' for trace in traces.values()),
            'supplemented_companies': sum(bool(trace.get('disputed_points')) for trace in traces.values()),
            'returned_models': sorted({call.get('returned_model', '') for call in calls}),
            'per_point': {str(point): {
                'correct': sum(pair[0] for pair in split.values()),
                'pass_recall': split['PASS'][0] / split['PASS'][1],
                'fail_recall': split['FAIL'][0] / split['FAIL'][1],
            } for point, split in counts.items()},
            'disagreements': disagreements,
        }
        entries.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k not in ['per_point', 'disagreements']},
                         ensure_ascii=False), flush=True)
    metrics_path = reports / 'metrics.json'
    previous = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else []
    replaced = {entry['label'] for entry in entries}
    combined = [entry for entry in previous if entry['label'] not in replaced] + entries
    metrics_path.write_text(json.dumps(combined, indent=2) + '\n', encoding='utf-8')
    fields = ['label', 'companies', 'correct', 'total', 'balanced_accuracy', 'pass_recall',
              'fail_recall', 'seconds', 'model_calls', 'input_tokens', 'output_tokens',
              'interviews', 'failed_companies', 'degraded_companies']
    with (reports / 'benchmark.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(combined)


if __name__ == '__main__':
    main()
