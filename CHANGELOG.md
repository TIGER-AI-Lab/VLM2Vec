# Changelog

All notable changes to this project will be documented in this file.

[bug-fix]: https://img.shields.io/badge/BUG%20FIX-blue
[new-feature]: https://img.shields.io/badge/NEW%20FEATURE-brightgreen
[new-release]: https://img.shields.io/badge/NEW%20RELEASE-orange

## ![new-feature] 2026-03-27 (Evaluation Consolidation & Reporting)
- **Column Ordering for Reports**: Updated `export_evals_to_csv.py` to place `VLM2Vec-v2_0` as the leftmost column in the final consolidated report.
- **Submitted EXP_04_1 evaluations**: Launched evaluations for `EXP_04_1` checkpoint-5000 (Jobs 9448-9473) on the Slurm cluster.
- **Consolidated EXP_06_1 evaluations**: Verified and exported consolidated results for `EXP_06_1` checkpoint-5000.

## ![bug-fix] 2026-03-26 (Stabilizing OLM2Vec Pipeline)
- **Bypassed broken `torchcodec`**: Renamed the `torchcodec` package folder to bypass the PyTorch C++ ABI mismatch error.
- **Pure Python Audio Fallback**: Implemented a fallback in `src/data/collator/train_collator_omni.py` to use Python’s standard `wave` module when `torchaudio.load()` throws `ImportError` for byte streams!
- **Stabilized 4 suites execution**: Re-launched all suites using global batch size 256 (`per_device_train_batch_size=8`, 4 nodes). Verified Suite 01 (`9340`) reached training loop steps 10+.

## ![bug-fix] 2025-11-03
- Fixed the issue in ```ViDoSeek-page``` and ```MMLongBench-page```. More details in this [issue](https://github.com/TIGER-AI-Lab/VLM2Vec/issues/167).

## ![new-release] 2025-08-08
- Release the raw video files [here](https://huggingface.co/datasets/TIGER-Lab/MMEB_Raw_Video). Please note that raw videos are not required for MMEB evaluation — video frames are all you need. We provide the raw videos only in case they are useful for your own purposes. Please refer to our [main data repository](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2) for more instructions.

## ![bug-fix] 2025-07-31
- Updated the MomentSeeker task by deduplicating some test cases. More details in this [issue](https://github.com/TIGER-AI-Lab/VLM2Vec/issues/123#issuecomment-3141653760).

## ![new-release] 2025-06-03

### Released v2.0.0.
- **Expanded Modality Support**: VLM2Vec now supports unified training and evaluation on three modalities: images, videos, and visual documents.
- **VLM2Vec v2**: Initial release of the V2 model and framework.
- **MMEB-v2 Benchmark**: Introduced a new comprehensive benchmark for evaluating performance across all supported modalities.

## ![new-release] 2025-02-11

### Released v1.1.0.
- Refactored the sub-batch splitting logic within GradCache and the VLM processor to simplify future extensions.
