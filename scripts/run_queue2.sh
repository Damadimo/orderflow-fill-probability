#!/bin/zsh
# Second queue: waits for the running B4 job, then trains the OFT variants (4 epochs, more regularisation).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while pgrep -f "tag b4_snapshot" > /dev/null; do sleep 15; done
run() { local tag=$1; shift; echo "=== $tag $(date)"; $PY scripts/train.py --tag $tag "$@" > logs/$tag.log 2>&1; tail -1 logs/$tag.log; }
run oft_nopre --model oft --epochs 4 --dropout 0.15 --wd 0.05
run oft_pre   --model oft --pretrain-epochs 1 --epochs 4 --dropout 0.15 --wd 0.05
echo "=== queue done $(date)"
