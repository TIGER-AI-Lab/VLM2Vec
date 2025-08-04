# Changelog

All notable changes to this project will be documented in this file.

[bug-fix]: https://img.shields.io/badge/BUG%20FIX-blue
[new-feature]: https://img.shields.io/badge/NEW%20FEATURE-brightgreen
[new-release]: https://img.shields.io/badge/NEW%20RELEASE-orange


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
