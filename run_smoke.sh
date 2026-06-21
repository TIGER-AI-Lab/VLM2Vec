set -e
cd /rmeng_data/projects/embed/VLM2Vec-olm2vec
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
MODEL=/rmeng_data/data/vlm2vec/models/omni-embed-nemotron-3b
CUDA_VISIBLE_DEVICES=0 /rmeng_data/envs/vlm2vec/bin/python eval.py \
  --pooling mean --normalize true \
  --per_device_eval_batch_size 8 \
  --model_backbone nvomniembed \
  --model_name "$MODEL" \
  --processor_name "$MODEL" \
  --dataset_config experiments/public/eval/smoke.yaml \
  --encode_output_path exps/smoke/ \
  --data_basedir /rmeng_data/data/vlm2vec/MMEB-V3-evalroot
