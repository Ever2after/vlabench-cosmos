export VLABENCH_ROOT="$(cd "$(dirname "$0")/VLABench" && pwd)"

python3 scripts/evaluate_policy.py \
  --policy cosmos_policy \
  --host localhost \
  --port 8777 \
  --tasks insert_flower \
  --n-episode 10 \
  --eval-track track_3_common_sense \
  --replanstep 4 \
  --save-dir logs/cosmos_policy_eval2 \
  --visualization