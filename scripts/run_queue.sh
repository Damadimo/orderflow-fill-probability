#!/bin/zsh
# Sequential training queue (one model at a time on the Apple GPU). Logs in logs/<tag>.log.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { local tag=$1; shift; echo "=== $tag $(date)"; $PY scripts/train.py --tag $tag "$@" > logs/$tag.log 2>&1; tail -1 logs/$tag.log; }

run b4_snapshot --model snap --epochs 6
run oft_nopre   --model oft --epochs 6
run oft_pre     --model oft --pretrain-epochs 1 --epochs 6
echo "=== queue done $(date)"
