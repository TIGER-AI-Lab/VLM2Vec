#!/bin/bash
# Submit all 26 task groups for VLM2Vec-V2.0 to Slurm sequentially

EXEC_DIR="/rmeng_data/projects/embed/VLM2Vec/experiments_private/olm2vec/eval"
CONFIG_DIR="$EXEC_DIR/configs/task_group"

cd "$EXEC_DIR" || exit

echo "🗑️ Clearing any existing eval jobs on the partition..."
scancel -u $USER -p a3 -n eval_ || true

echo "🚀 Submitting All 26 Task Group Evaluations for VLM2Vec-V2.0"

for yaml_file in "$CONFIG_DIR"/*.yaml; do
    [ -e "$yaml_file" ] || continue
    group_name=$(basename "$yaml_file" .yaml)
    
    echo "📡 Submitting Group: $group_name"
    
    sbatch --gres=gpu:8 \
           --exclude=slurm2-a3nodeset-0 \
           --job-name="eval_group_${group_name}" \
           --export=ALL,TARGET_MOD="$group_name" \
           --output="/rmeng_data/projects/embed/VLM2Vec/experiments_private/olm2vec/slurm_logs/eval/%j_eval_${group_name}.out" \
           --error="/rmeng_data/projects/embed/VLM2Vec/experiments_private/olm2vec/slurm_logs/eval/%j_eval_${group_name}.err" \
           eval_v2.0_groups.sbatch
done

echo "✅ All task groups submitted."
