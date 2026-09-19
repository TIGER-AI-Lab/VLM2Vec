# Qwen3-VL-Embedding-8B on MMEB-v1/v2: reproducing the leaderboard

Our inference code scored Qwen3-VL-Embedding-8B 7-9 points below the MMEB-V2 leaderboard on all
three suites. The checkpoint, the data and `transformers` were not the cause; the inference code
was, in six places. Qwen's own evaluation code, run unmodified on our node and data, reproduces
the leaderboard to 0.1-0.3 on every dataset tried; after the fixes our code produces the same
embeddings as theirs (per-item cosine 0.996-1.002) and the same scores.

| suite | leaderboard | before | after | spread after |
|---|--:|--:|--:|---|
| Image (36, hit@1) | 80.12 | 73.02 | **80.12** | 34 of 36 within 0.5, max 0.6 |
| Video (18, hit@1) | 67.15 | 58.54 | **67.62** | 14 of 18 within 1.0; +3.4 HMDB51, +3.2 ActivityNetQA, -0.8 DiDeMo |
| VisDoc (24, ndcg@5) | 82.36 | 73.08 (27 sets) | **82.03** | 21 of 24 within 1.0; -2.5 on a 52-query set (one rank flip is ~1.9) |

Per-dataset tables: `results_qwen3vl8b.md`. Raw score files of the final runs:
`scores/qwen3vl8b/`. Leaderboard reference: `leaderboard/Qwen3-VL-Embedding-8B.json`
(`scores/Qwen3-VL-Embedding-8B.json` in the TIGER-Lab/MMEB-Leaderboard space; image/video are
means of hit@1, visdoc of ndcg_linear@5).

## Where the gap came from

Each row adds one fix to the previous one; means are over the datasets present in every run.

| step | image (36) | video (11 of 18) | visdoc (22 of 24) |
|---|--:|--:|--:|
| before | 73.02 | 58.38 | 75.99 |
| task instruction in the system turn (#2) | +3.81 | +2.39 | -1.33 |
| candidates without instruction (#3) | -0.12 | | +1.20 |
| t2i query routing, pixel budget, video path (#3, #6) | +1.14 | | |
| videos encoded as video, 64 frames (#4) | | +6.36 | |
| raw whitespace in texts (#6) | +0.06 | | |
| pool the normed hidden state (#1) | +2.21 | +2.98 | +6.96 |
| after | 80.12 | 70.11 | 82.82 |
| leaderboard, same datasets | 80.12 | 69.53 | 83.09 |

The visdoc row for #2 is negative because that run also routed the *candidate* instruction into
the system turn, which #3 undid. The pooling fix is worth most on visdoc, whose candidates are
1,800-token page images, and least on image.

### 1. The pooled state was taken before the final norm

`MMEBModel.encode_input` ran `Qwen3VLForConditionalGeneration(..., output_hidden_states=True)` and
pooled `outputs.hidden_states[-1]`. Up to transformers 4.55 that tensor was the output after the
final RMSNorm (the manual layer loop appended the normed state last). From 4.56 hidden states are
recorded by the `check_model_inputs` hooks as decoder-layer outputs, and the normed
`last_hidden_state` is swapped in for the last entry only when the output object has that field.
The base `Qwen3VLModel` output does; the CausalLM output (`logits`, `hidden_states`, ...) does not.
Through the CausalLM wrapper, `hidden_states[-1]` is therefore the last decoder layer's output
*before* the norm. The same line of code changed meaning with the `transformers` upgrade.

The final RMSNorm is `x / rms(x) * gamma` with a learned 4096-dim `gamma`. L2-normalising the
embedding for cosine similarity (done in training and inference by both pipelines) fixes the
length only; `gamma` rescales each dimension and changes the direction. Cosine between the
pre-norm and post-norm embeddings is 0.87-0.97. Which one is right is set by training: Qwen's
`Qwen3VLForEmbedding` pools `last_hidden_state` (post-norm), and the checkpoint's
Sentence-Transformers config pools the transformer output (also post-norm).

Evidence, identical tensors through both models on one GPU: 749/749 weight tensors identical,
position ids identical, flash-attention-2 on both, per-layer last-token cosine >= 0.9998 at every
layer; the CausalLM-wrapper read differs from the normed state at 0.87-0.97; after the fix
0.996-1.002 (Qwen's own batched-vs-single runs differ by up to 0.007). Per query the error looked
like noise: where their run was right and ours wrong, our gold item sat at rank 2-3. The pre-norm
activations carry massive outlier dimensions (max abs differences ~13,000 in layers 18-33 while
cosines stayed 0.9999); the norm's `gamma` tames them, pooling them raw lets a few dimensions
dominate every dot product.

Fix (`src/model/model.py`, fallback branch of `encode_input`): for `qwen3_vl` call
`self.encoder.model(...)` and pool `last_hidden_state`. Every other backbone this code evaluates
through a CausalLM wrapper on transformers >= 4.56 (Qwen2-VL / Qwen2.5-VL VLM2Vec-V2 checkpoints,
LLaVA-Next, Phi3V) has the same exposure and was not checked here.

### 2. The task instruction was not in the system turn

Qwen3-VL-Embedding takes the task instruction as the system message. `Qwen3_VL_Embedding_process_fn`
hardcoded the system turn to `"Represent the user's input."` and left the instruction in the user
text. `process_input_text` now marks the instruction/text boundary with a sentinel for this backbone
and the process_fn routes the instruction to the system turn. `image_t2i_eval.py` built its query by
string concatenation and bypassed this; routed as well.

### 3. Candidates carried an instruction; Qwen's do not

In Qwen's evaluation code every `TASK_INST_TGT` / `tgt_inst` is commented out: document pages,
video clips, labels and captions are encoded under the default system prompt with no task text.
Our parsers glued the target instruction into every candidate, and `image_i2i_vg` / `image_t2i`
put the caption into the instruction slot. The collator now passes `encode_side`; on the candidate
side the process_fn drops the instruction and keeps the content; the two parsers pass the caption
as text. RefCOCO-Matching 82.8 -> 92.4 (leaderboard 91.1), WebQA 86.4 -> 89.5, VisDial 68 -> 81.

### 4. Videos: a bug on our side, plus Qwen's evaluation setting

Bug: the eval collator delivers video frames as a list under `'images'` with the video token in
the text; the Qwen3 process_fn only looked at a `'videos'` key nobody sets, so every video was
encoded as N independent images (no temporal merging, no video pixel budget).

Setting: Qwen evaluates with `num_frames: 64` (`num_video_frames: 64` for moment retrieval), a
`total_pixels` budget of 7,864,320 shared across the frames, `do_resize=False`, and passes the
frame metadata with `do_sample_frames=False`. VLM2Vec's default is 8 frames. The leaderboard
number is defined at Qwen's setting, so `mmeb_v2_video_64f.yaml` uses it.

The two are measured together (+6.36 on 11 datasets). An earlier ablation that raised 8 -> 32
frames while still encoding them as images gained +1.7 on 8 datasets, so most of the +6.4 is the
encoding, not the frame count. Six video parsers sized their frame lists by `num_frames` and
asserted on videos with fewer saved frames; they now use the actual list.

### 5. The leaderboard's dataset sets differ from ours

- VisDoc is scored over 24 datasets: the three English-only ViDoRe-v2 sets are absent and
  `MMLongBench-page` / `ViDoSeek-page` are the `VLM2Vec/*-page-fixed` releases.
  `mmeb_v2_visdoc.yaml` is that set. The fixed data is staged at
  `/rmeng_data/data/vlm2vec/hf_fixed/`, symlinked into `MMEB-V3-eval/visdoc-tasks/data/`, with
  PNGs pre-extracted to `visdoc-tasks/images/<name>-fixed/`.
- MomentSeeker on the leaderboard has 1,602 rows (`VLM2Vec/MomentSeeker`); our
  `momentseeker_1k6.jsonl` was a symlink to the 1,800-row file. It is now the 1,602-row subset
  (old link kept as `.bak_1800link`). MomentSeeker: 46.7 -> 54.4 (leaderboard 54.9).

### 6. Smaller items

- `qwen-vl-utils` 0.0.8 does not accept `image_patch_size`; the call fell back to the Qwen2-VL
  28-px grid and every image was resized twice. 0.0.14 is side-installed at
  `/rmeng_data/envs/qwen3vl_extra` and passed via `EXTRA_PYTHONPATH`. +0.2 on image.
- The text cleaner collapsed all whitespace; Qwen feeds raw text. VisDial's newline-separated
  dialogue and the WebQA / EDIS / MSCOCO_i2t captions tokenise differently. Fixed (VisDial +1.7).
- Pixel budget 4096 / 1,843,200 per image as in Qwen's code: no measurable effect on MMEB image
  (median image is 168k px) or visdoc; kept for fidelity. Passed as `RESIZE_MIN_PIXELS` /
  `RESIZE_MAX_PIXELS`, applied per content item so `qwen_vl_utils` is the only resize.
- Not mirrored: our hardcoded visdoc and video query instructions end with `:`, Qwen's with `.`
  (the instructions stored in the MMEB image data end with `:` in both). One token in the system
  turn; left as is because the constants are shared with the other backbones. The visdoc residual
  of -0.33 is within what this could account for.

## Qwen's evaluation settings, in one place

From `scripts/qwen3_vl_embedding.py` (shipped with the checkpoint) and
github.com/QwenLM/Qwen3-VL-Embedding `src/evaluation/mmeb_v2`:

- instruction in the system turn; candidates with the default prompt `Represent the user's input.`
- images: `min_pixels` 4,096, `max_pixels` 1,843,200 per image (`image_patch_size=16`)
- video: 64 frames, `total_pixels` 7,864,320 per video, frame metadata passed, `do_sample_frames=False`
- processor: `padding_side='right'`, `do_resize=False`, `truncation=True`, `max_length=8192`
- model: base `Qwen3VLModel`, pool the last non-pad token of `last_hidden_state`, L2-normalise;
  bf16, flash-attention-2; `transformers>=4.57`, `qwen-vl-utils>=0.0.14`
- data: `ziyjiang/MMEB_Test_Instruct` parquets, 24 visdoc sets with `*-page-fixed`,
  `VLM2Vec/MomentSeeker` (1,602 rows), per-device batch 16

## Checkpoint and upstream

The HF checkpoint has 9 commits; the four safetensors shards, `config.json`,
`preprocessor_config.json`, `chat_template.jinja` and `tokenizer_config.json` are byte-identical
from the 2026-01-07 upload through our 2026-04-16 snapshot (`2c45655`). Only the README, the
reference script (a refactor) and the Sentence-Transformers files changed. `transformers` is 4.57.1,
as `config.json` declares. TIGER-AI-Lab/VLM2Vec `main` added Qwen3-VL on 2026-03-16 (`c0e164e`)
with a process_fn identical to the one we started from, so it carries items 1-4 as well.

## How the causes were found

1. Qwen's published evaluation code (leaderboard thread #97 -> the GitHub repo above; local clone
   `/rmeng_data/projects/embed/Qwen3-VL-Embedding-ref`) was diffed against our parsers, collator,
   processor call and configs. Items 2-5 come from that diff.
2. The same parquet rows were run through their parser + `format_model_input` +
   `_preprocess_inputs` and through ours; `input_ids`, `image_grid_thw` and `pixel_values` are
   identical on CIRR, Wiki-SS-NQ, ImageNet-1K, FashionIQ, VisDial and ViDoRe_arxivqa. This proved
   preprocessing identical and left only the model side.
3. Their code was run unmodified on our node and data (`/rmeng_data/exps/vlm2vec/qref_run/`):
   CIRR 75.0 / Wiki-SS-NQ 88.2 / ImageNet-1K 81.9 / VisDial 87.6 and ViDoRe_arxivqa 86.9 /
   ViDoRe_docvqa 54.1 / VisRAG_ArxivQA 88.4 / VisRAG_MP-DocVQA 89.2, against leaderboard
   74.9 / 87.9 / 81.9 / 87.6 and 87.0 / 54.0 / 88.2 / 89.3. That cleared checkpoint, data,
   `transformers` and hardware at once.
4. Identical tensors through both model wrappers, compared per layer, located item 1.

Hypotheses tested and rejected: missing instructions in the V3 data (+0.00), pixel budget
(+0.00 image, -0.03 visdoc), candidate-pool dedup (pools already unique), DDP gather/trim
misalignment, the eval loop's per-grid bucketing (identical scores with it bypassed), a leftover
rotary-dtype env switch (not consumed in this branch), modified `transformers` files (pristine).

## Reproducing

```bash
EVAL=experiments/mmebv3_reproduction/run_eval.sbatch
export MODEL_PATH=/rmeng_data/data/vlm2vec/models/Qwen3-VL-Embedding-8B MODEL_BACKBONE=qwen3_vl \
  POOLING=last ENV_PYTHON=/rmeng_data/envs/qwen3vl/bin/python \
  DATA_BASEDIR=/rmeng_data/data/vlm2vec/MMEB-V2-eval EXTRA_PYTHONPATH=/rmeng_data/envs/qwen3vl_extra \
  RESIZE_MIN_PIXELS=4096 RESIZE_MAX_PIXELS=1843200 NPROC=8
C=experiments/mmebv3_reproduction/configs
DATASET_CONFIG=$C/mmeb_v1_image.yaml     OUTPUT_PATH=<out>/image  BATCH=8 sbatch $EVAL
DATASET_CONFIG=$C/mmeb_v2_visdoc.yaml    OUTPUT_PATH=<out>/visdoc BATCH=8 sbatch $EVAL
DATASET_CONFIG=$C/mmeb_v2_video_64f.yaml OUTPUT_PATH=<out>/video  BATCH=4 \
  VIDEO_MAX_FRAMES=64 VIDEO_FRAME_SIZE=0 sbatch $EVAL
python experiments/mmebv3_reproduction/compare_to_leaderboard.py \
  --lb docs/mmebv2_reproduction/leaderboard/Qwen3-VL-Embedding-8B.json \
  --image <out>/image --video <out>/video --visdoc <out>/visdoc
```

`VIDEO_MAX_FRAMES=64 VIDEO_FRAME_SIZE=0` is required: the eval collator caps candidate-side
videos at `--video_max_frames` (default 8) and squares each frame to `--video_frame_size`
(default 224), which would put the retrieval sets back at 8 frames of 224x224.

`MMEB-V2-eval` is an overlay root: `image-tasks/<ds>` symlinks to the `ziyjiang/MMEB_Test_Instruct`
parquets, `image-tasks/MMEB` and `video-tasks` / `visdoc-tasks` symlink into `MMEB-V3-eval`.
Run times on one 8xH100 node: image 27 min, visdoc 43 min, video (64 frames) 2 h 50 min.

## Running on current `main`

The numbers above were produced on a branch based on `0a28744`. The fix set was re-applied
onto `main` at `8713911` (2026-09-19) and re-run: image 36/36 datasets bit-identical, visdoc
23/24 identical (VisRAG_PlotQA +0.11, one query out of 863), video identical on DiDeMo and
ActivityNetQA. Three things on `main` differ from that older base and matter for this recipe:

1. **Candidate-side video frames.** The rewritten eval collator downsamples candidate videos to
   `--video_max_frames` (default 8) and squares each frame to `--video_frame_size` (default
   224). `run_eval.sbatch` forwards `VIDEO_MAX_FRAMES` / `VIDEO_FRAME_SIZE`; the reproduce
   block above sets them.
2. **Short clips are padded.** `63ee24a` (2026-06-22) changed `process_video_frames` in
   `src/utils/vision_utils/vision_utils.py` to always call `sample_frames`, which pads a clip
   with fewer than `num_frames` saved frames by repeating its last frame. Qwen's reference code
   (and therefore the leaderboard) uses the saved frames as they are. On our frame dumps this
   touches Charades-STA (100% of clips under 64 frames, median 11), HMDB51 (98%, median 16),
   SmthSmthV2 (89%, median 48), MSR-VTT (84%, median 41) and MVBench (15%); the other 13 sets
   are at most 3%. Measured on `main`: HMDB51 84.50 padded vs 86.80 with the reference
   behaviour (the number in the tables above), MSR-VTT 59.10 vs 59.50, MVBench 68.17 vs 68.27.
   The branch leaves `main`'s behaviour in place; to reproduce the leaderboard on those sets,
   restore the `num_frames <= len(frames)` guard in `process_video_frames`.
3. **VisRAG loader memory.** `main` maps the VisRAG corpus with `num_proc=4, batch_size=1024`.
   With 8 ranks that is 32 workers decoding 1024 pages each; the node (512 GB) OOM-killed on
   VisRAG_ArxivQA. The branch restores the previous `num_proc=1, batch_size=256`.

## Files

- `README.md` - this document
- `results_qwen3vl8b.md` - per-dataset before / after / leaderboard tables
- `scores/qwen3vl8b/{image,video,visdoc}/` - `*_score.json` of the final runs
- `leaderboard/Qwen3-VL-Embedding-8B.json` - the leaderboard's per-dataset file
- code: `src/model/model.py`, `src/model/processor.py`, `src/data/collator/eval_collator.py`,
  `src/data/eval_dataset/{image_t2i_eval,image_i2i_vg_dataset,didemo,msrvtt,msrvtt_dataset_test,msvd,vatex,youcook2}*.py`,
  `src/constant/dataset_hf*_path.py`; configs and `compare_to_leaderboard.py` under
  `experiments/mmebv3_reproduction/`
