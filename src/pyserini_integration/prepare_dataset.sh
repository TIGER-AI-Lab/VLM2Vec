#!/usr/bin/env bash

set -ex

LOCAL_DIR="/mnt/users/s8sharif/MMEB-V2" # <--- change this to the desired local directory
OUTPUT_DIR="$LOCAL_DIR/visdoc-tasks/pyserini"
#python src/pyserini_integration/download_visdoc.py --local-dir $LOCAL_DIR

# tar -zxf $LOCAL_DIR/visdoc-tasks/visdoc-tasks.data.tar.gz -C $LOCAL_DIR/visdoc-tasks
# tar -zxf $LOCAL_DIR/visdoc-tasks/visdoc-tasks.images.tar.gz -C $LOCAL_DIR/visdoc-tasks

python src/pyserini_integration/save_pyserini_data.py \
    --yaml_file src/pyserini_integration/visdoc.yaml \
    --data_basedir $LOCAL_DIR/visdoc-tasks \
    --output_dir $OUTPUT_DIR \
    --num_workers 4