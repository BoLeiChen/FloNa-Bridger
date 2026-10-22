import json
from pathlib import Path
import subprocess
import sys


def test_full_benchmark_rejects_partial_results(tmp_path):
    root = Path(__file__).parents[1]
    (tmp_path / 'experiments').mkdir()
    (tmp_path / 'experiments/pointgoal_v3_results.json').write_text((root / 'experiments/pointgoal_v3_results.json').read_text())
    tasks = json.loads((root / 'experiments/pointgoal_v3_results.json').read_text())['tasks']
    outputs = tmp_path / 'results/exps/partial/trajectory'
    outputs.mkdir(parents=True)
    (outputs / tasks[0]['filename']).write_text('0 0 0 0 0\n')
    (tmp_path / 'config.yaml').write_text('data_root: missing-data\n')
    result = subprocess.run([sys.executable, str(root / 'results/evaluate.py'), '--config', 'config.yaml', '--exp_name', 'partial', '--require-full-benchmark'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert '499 missing' in result.stderr
    assert not (tmp_path / 'evaluation_summary.csv').exists()
