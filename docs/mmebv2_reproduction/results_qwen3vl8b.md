# Qwen3-VL-Embedding-8B on MMEB-v1/v2: per-dataset results

before fix = the inference code before any change; PR branch = branch `rmeng/qwen3vl-mmebv2-fix` (upstream `main` plus the fixes; identical to the final run on the original branch whose score files are in `scores/qwen3vl8b/`); leaderboard = `leaderboard/Qwen3-VL-Embedding-8B.json`. Image/video: hit@1; visdoc: ndcg_linear@5. Blank = not in that run; a column mean is over the sets it has.

## Summary by modality

| modality | n | before fix | PR branch (main + fixes) | leaderboard | PR branch − leaderboard |
|---|--:|--:|--:|--:|--:|
| image | 36 | 73.02 | 80.12 | 80.12 | -0.01 |
| video | 18 | 58.54 | 67.62 | 67.15 | +0.47 |
| visdoc | 24 | 75.99 (n=22) | 82.04 | 82.36 | -0.33 |
| all datasets (plain mean) | 78 | 70.45 (n=76) | 77.82 | 77.82 | +0.01 |

## image (36 datasets): PR branch 80.12 vs leaderboard 80.12

| dataset | before fix | PR branch (main + fixes) | leaderboard | PR branch − leaderboard |
|---|--:|--:|--:|--:|
| A-OKVQA | 67.20 | 72.60 | 72.90 | -0.30 |
| CIRR | 56.60 | 74.70 | 74.90 | -0.20 |
| ChartQA | 70.20 | 74.60 | 74.30 | +0.30 |
| Country211 | 25.10 | 27.60 | 27.30 | +0.30 |
| DocVQA | 95.10 | 96.20 | 96.30 | -0.10 |
| EDIS | 88.60 | 96.40 | 96.30 | +0.10 |
| FashionIQ | 32.00 | 44.00 | 44.50 | -0.50 |
| GQA | 88.50 | 92.50 | 92.60 | -0.10 |
| HatefulMemes | 67.50 | 77.40 | 77.50 | -0.10 |
| ImageNet-1K | 75.10 | 82.20 | 81.90 | +0.30 |
| ImageNet-A | 61.70 | 77.30 | 77.10 | +0.20 |
| ImageNet-R | 91.80 | 94.10 | 93.90 | +0.20 |
| InfographicsVQA | 81.60 | 87.90 | 88.40 | -0.50 |
| MSCOCO | 75.50 | 86.00 | 86.30 | -0.30 |
| MSCOCO_i2t | 73.50 | 79.60 | 79.10 | +0.50 |
| MSCOCO_t2i | 74.40 | 80.90 | 81.10 | -0.20 |
| N24News | 63.80 | 80.40 | 80.60 | -0.20 |
| NIGHTS | 68.10 | 72.90 | 72.70 | +0.20 |
| OK-VQA | 74.70 | 78.00 | 77.80 | +0.20 |
| OVEN | 70.20 | 79.10 | 79.20 | -0.10 |
| ObjectNet | 78.40 | 80.40 | 79.90 | +0.50 |
| Place365 | 38.20 | 48.10 | 47.60 | +0.50 |
| RefCOCO | 93.90 | 95.80 | 95.70 | +0.10 |
| RefCOCO-Matching | 93.60 | 90.90 | 91.10 | -0.20 |
| SUN397 | 65.00 | 82.90 | 82.70 | +0.20 |
| ScienceQA | 75.30 | 81.00 | 81.00 | +0.00 |
| TextVQA | 90.20 | 92.60 | 92.80 | -0.20 |
| VOC2007 | 84.80 | 93.30 | 93.40 | -0.10 |
| VisDial | 66.50 | 87.40 | 87.60 | -0.20 |
| Visual7W | 65.90 | 70.60 | 70.90 | -0.30 |
| Visual7W-Pointing | 87.90 | 95.90 | 96.10 | -0.20 |
| VisualNews_i2t | 81.90 | 85.10 | 85.70 | -0.60 |
| VisualNews_t2i | 75.60 | 81.20 | 81.10 | +0.10 |
| VizWiz | 59.30 | 64.80 | 64.40 | +0.40 |
| WebQA | 90.00 | 91.80 | 91.80 | +0.00 |
| Wiki-SS-NQ | 81.10 | 88.00 | 87.90 | +0.10 |
| **MEAN** | **73.02** | **80.12** | **80.12** | **-0.01** |

## video (18 datasets): PR branch 67.62 vs leaderboard 67.15

| dataset | before fix | PR branch (main + fixes) | leaderboard | PR branch − leaderboard |
|---|--:|--:|--:|--:|
| ActivityNetQA | 80.00 | 83.30 | 80.10 | +3.20 |
| Breakfast | 39.49 | 64.20 | 64.67 | -0.47 |
| Charades-STA | 26.55 | 33.43 | 34.11 | -0.68 |
| DiDeMo | 45.22 | 65.24 | 66.04 | -0.80 |
| EgoSchema | 65.20 | 68.40 | 69.00 | -0.60 |
| HMDB51 | 75.30 | 86.80 | 83.40 | +3.40 |
| K700 | 51.90 | 67.50 | 67.60 | -0.10 |
| MSR-VTT | 50.60 | 59.50 | 58.20 | +1.30 |
| MSVD | 72.69 | 76.57 | 75.67 | +0.90 |
| MVBench | 64.98 | 68.27 | 66.90 | +1.37 |
| MomentSeeker | 46.67 | 54.37 | 54.93 | -0.56 |
| NExTQA | 72.48 | 75.74 | 76.19 | -0.45 |
| QVHighlight | 71.28 | 80.06 | 79.22 | +0.84 |
| SmthSmthV2 | 73.50 | 81.50 | 81.20 | +0.30 |
| UCF101 | 85.50 | 95.30 | 95.10 | +0.20 |
| VATEX | 44.95 | 54.98 | 54.87 | +0.11 |
| Video-MME | 55.63 | 62.44 | 62.63 | -0.19 |
| YouCook2 | 31.77 | 39.57 | 38.85 | +0.72 |
| **MEAN** | **58.54** | **67.62** | **67.15** | **+0.47** |

## visdoc (24 datasets): PR branch 82.04 vs leaderboard 82.36

| dataset | before fix | PR branch (main + fixes) | leaderboard | PR branch − leaderboard |
|---|--:|--:|--:|--:|
| MMLongBench-doc | 53.87 | 58.40 | 58.29 | +0.11 |
| MMLongBench-page-fixed |  | 59.79 | 60.29 | -0.50 |
| ViDoRe_arxivqa | 79.41 | 86.56 | 87.00 | -0.44 |
| ViDoRe_biomedical_lectures_v2_multilingual | 60.80 | 71.41 | 71.59 | -0.18 |
| ViDoRe_docvqa | 46.26 | 53.79 | 54.02 | -0.23 |
| ViDoRe_economics_reports_v2_multilingual | 50.60 | 66.02 | 67.18 | -1.16 |
| ViDoRe_esg_reports_human_labeled_v2 | 54.91 | 68.86 | 71.39 | -2.53 |
| ViDoRe_esg_reports_v2_multilingual | 50.75 | 69.00 | 69.28 | -0.28 |
| ViDoRe_infovqa | 85.58 | 90.96 | 90.86 | +0.10 |
| ViDoRe_shiftproject | 80.07 | 84.35 | 84.14 | +0.21 |
| ViDoRe_syntheticDocQA_artificial_intelligence | 96.42 | 98.52 | 99.26 | -0.74 |
| ViDoRe_syntheticDocQA_energy | 88.86 | 94.04 | 94.41 | -0.37 |
| ViDoRe_syntheticDocQA_government_reports | 93.82 | 97.89 | 98.02 | -0.13 |
| ViDoRe_syntheticDocQA_healthcare_industry | 95.68 | 98.02 | 97.79 | +0.23 |
| ViDoRe_tabfquad | 94.74 | 97.01 | 96.67 | +0.34 |
| ViDoRe_tatdqa | 59.67 | 69.95 | 69.97 | -0.02 |
| ViDoSeek-doc | 84.69 | 86.18 | 86.10 | +0.08 |
| ViDoSeek-page-fixed |  | 86.99 | 88.38 | -1.39 |
| VisRAG_ArxivQA | 78.41 | 87.99 | 88.21 | -0.22 |
| VisRAG_ChartQA | 87.23 | 88.93 | 88.34 | +0.59 |
| VisRAG_InfoVQA | 90.48 | 94.67 | 94.75 | -0.08 |
| VisRAG_MP-DocVQA | 80.48 | 88.79 | 89.27 | -0.48 |
| VisRAG_PlotQA | 66.00 | 73.98 | 74.80 | -0.82 |
| VisRAG_SlideVQA | 93.09 | 96.80 | 96.73 | +0.07 |
| **MEAN** | **75.99 (n=22)** | **82.04** | **82.36** | **-0.33** |
