#!/bin/zsh
# Token ablations with the same recipe as oft_nopre (B4 re-run b4_matched already done by the first version of this queue).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { local tag=$1; shift; echo "=== $tag $(date)"; $PY scripts/train.py --tag $tag "$@" > logs/$tag.log 2>&1; tail -1 logs/$tag.log; }
run oft_noqueue  --model oft --epochs 4 --dropout 0.15 --wd 0.05 --no-queue
run oft_noevents --model oft --epochs 4 --dropout 0.15 --wd 0.05 --no-events
echo "=== queue3 done $(date)"
