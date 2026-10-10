#!/usr/bin/env bash
#SBATCH --job-name=tadse-recovered-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --partition=voltagepark
#SBATCH --qos=jit
#SBATCH --account=yejinchoi
set -euo pipefail
CODE="${1:?immutable evaluator checkout}"
WORK="${2:?primary checkout}"
PLAN="${3:?evaluation manifest JSON}"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export HF_HOME="$WORK/artifacts/hf" PYTHONUNBUFFERED=1
cd "$CODE"
"$WORK/.venv/bin/python" - "$CODE" "$WORK" "$PLAN" <<'PY'
import json, subprocess, sys
from pathlib import Path
code, work, plan = map(Path, sys.argv[1:])
m = json.loads(plan.read_text())
for run in m['runs']:
    output = Path(run['output'])
    if output.exists():
        raise FileExistsError(f'Preserving existing evaluation: {output}')
    subprocess.run([str(work/'.venv/bin/python'), str(code/'scripts/evaluate_recovered.py'),
        '--checkpoint', run['checkpoint'], '--data-root', str(work/'data/massive'),
        '--clinc-splits-json', str(work/'artifacts/sources/data_oos_plus.json'),
        '--output', str(output)], check=True)
results = [{**r, 'results': json.loads(Path(r['output']).read_text())} for r in m['runs']]
assert all(r['results']['complete'] for r in results)
summary = sorted([{'name':r['name'], 'output':r['output'],
    'mean_accuracy':r['results']['mean_accuracy'],
    'tasks': {k: {'accuracy':v['accuracy'], 'test_count':v['test_count']} for k,v in r['results']['tasks'].items()}}
    for r in results], key=lambda r:r['mean_accuracy'], reverse=True)
plan.with_suffix('.results.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
PY
