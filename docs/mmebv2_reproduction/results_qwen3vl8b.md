# Qwen3-VL-Embedding-8B on MMEB-v1/v2: per-dataset results

Ours = final run with all fixes (`scores/qwen3vl8b/`); original = the inference code before any fix; LB = leaderboard file `leaderboard/Qwen3-VL-Embedding-8B.json`. Image/video: hit@1; visdoc: ndcg_linear@5.


## image (36 datasets): ours 80.12 vs LB 80.12

| dataset | original | ours | LB | ours − LB |
|---|--:|--:|--:|--:|
| A-OKVQA | 67.2 | 72.6 | 72.9 | -0.3 |
| CIRR | 56.6 | 74.7 | 74.9 | -0.2 |
| ChartQA | 70.2 | 74.6 | 74.3 | +0.3 |
| Country211 | 25.1 | 27.6 | 27.3 | +0.3 |
| DocVQA | 95.1 | 96.2 | 96.3 | -0.1 |
| EDIS | 88.6 | 96.4 | 96.3 | +0.1 |
| FashionIQ | 32.0 | 44.0 | 44.5 | -0.5 |
| GQA | 88.5 | 92.5 | 92.6 | -0.1 |
| HatefulMemes | 67.5 | 77.4 | 77.5 | -0.1 |
| ImageNet-1K | 75.1 | 82.2 | 81.9 | +0.3 |
| ImageNet-A | 61.7 | 77.3 | 77.1 | +0.2 |
| ImageNet-R | 91.8 | 94.1 | 93.9 | +0.2 |
| InfographicsVQA | 81.6 | 87.9 | 88.4 | -0.5 |
| MSCOCO | 75.5 | 86.0 | 86.3 | -0.3 |
| MSCOCO_i2t | 73.5 | 79.6 | 79.1 | +0.5 |
| MSCOCO_t2i | 74.4 | 80.9 | 81.1 | -0.2 |
| N24News | 63.8 | 80.4 | 80.6 | -0.2 |
| NIGHTS | 68.1 | 72.9 | 72.7 | +0.2 |
| OK-VQA | 74.7 | 78.0 | 77.8 | +0.2 |
| OVEN | 70.2 | 79.1 | 79.2 | -0.1 |
| ObjectNet | 78.4 | 80.4 | 79.9 | +0.5 |
| Place365 | 38.2 | 48.1 | 47.6 | +0.5 |
| RefCOCO | 93.9 | 95.8 | 95.7 | +0.1 |
| RefCOCO-Matching | 93.6 | 90.9 | 91.1 | -0.2 |
| SUN397 | 65.0 | 82.9 | 82.7 | +0.2 |
| ScienceQA | 75.3 | 81.0 | 81.0 | +0.0 |
| TextVQA | 90.2 | 92.6 | 92.8 | -0.2 |
| VOC2007 | 84.8 | 93.3 | 93.4 | -0.1 |
| VisDial | 66.5 | 87.4 | 87.6 | -0.2 |
| Visual7W | 65.9 | 70.6 | 70.9 | -0.3 |
| Visual7W-Pointing | 87.9 | 95.9 | 96.1 | -0.2 |
| VisualNews_i2t | 81.9 | 85.1 | 85.7 | -0.6 |
| VisualNews_t2i | 75.6 | 81.2 | 81.1 | +0.1 |
| VizWiz | 59.3 | 64.8 | 64.4 | +0.4 |
| WebQA | 90.0 | 91.8 | 91.8 | +0.0 |
| Wiki-SS-NQ | 81.1 | 88.0 | 87.9 | +0.1 |

## video (18 datasets): ours 67.62 vs LB 67.15

| dataset | original | ours | LB | ours − LB |
|---|--:|--:|--:|--:|
| ActivityNetQA | 80.0 | 83.3 | 80.1 | +3.2 |
| Breakfast | 39.5 | 64.2 | 64.7 | -0.5 |
| Charades-STA | 26.5 | 33.4 | 34.1 | -0.7 |
| DiDeMo | 45.2 | 65.2 | 66.0 | -0.8 |
| EgoSchema | 65.2 | 68.4 | 69.0 | -0.6 |
| HMDB51 | 75.3 | 86.8 | 83.4 | +3.4 |
| K700 | 51.9 | 67.5 | 67.6 | -0.1 |
| MSR-VTT | 50.6 | 59.5 | 58.2 | +1.3 |
| MSVD | 72.7 | 76.6 | 75.7 | +0.9 |
| MVBench | 65.0 | 68.3 | 66.9 | +1.4 |
| MomentSeeker | 46.7 | 54.4 | 54.9 | -0.6 |
| NExTQA | 72.5 | 75.7 | 76.2 | -0.5 |
| QVHighlight | 71.3 | 80.1 | 79.2 | +0.8 |
| SmthSmthV2 | 73.5 | 81.5 | 81.2 | +0.3 |
| UCF101 | 85.5 | 95.3 | 95.1 | +0.2 |
| VATEX | 45.0 | 55.0 | 54.9 | +0.1 |
| Video-MME | 55.6 | 62.4 | 62.6 | -0.2 |
| YouCook2 | 31.8 | 39.6 | 38.8 | +0.7 |

## visdoc (24 datasets): ours 82.03 vs LB 82.36

| dataset | original | ours | LB | ours − LB |
|---|--:|--:|--:|--:|
| MMLongBench-doc | 53.9 | 58.4 | 58.3 | +0.1 |
| MMLongBench-page-fixed | nan | 59.8 | 60.3 | -0.5 |
| ViDoRe_arxivqa | 79.4 | 86.6 | 87.0 | -0.4 |
| ViDoRe_biomedical_lectures_v2_multilingual | 60.8 | 71.4 | 71.6 | -0.2 |
| ViDoRe_docvqa | 46.3 | 53.8 | 54.0 | -0.2 |
| ViDoRe_economics_reports_v2_multilingual | 50.6 | 66.0 | 67.2 | -1.2 |
| ViDoRe_esg_reports_human_labeled_v2 | 54.9 | 68.9 | 71.4 | -2.5 |
| ViDoRe_esg_reports_v2_multilingual | 50.7 | 69.0 | 69.3 | -0.3 |
| ViDoRe_infovqa | 85.6 | 91.0 | 90.9 | +0.1 |
| ViDoRe_shiftproject | 80.1 | 84.4 | 84.1 | +0.2 |
| ViDoRe_syntheticDocQA_artificial_intelligence | 96.4 | 98.5 | 99.3 | -0.7 |
| ViDoRe_syntheticDocQA_energy | 88.9 | 94.0 | 94.4 | -0.4 |
| ViDoRe_syntheticDocQA_government_reports | 93.8 | 97.9 | 98.0 | -0.1 |
| ViDoRe_syntheticDocQA_healthcare_industry | 95.7 | 98.0 | 97.8 | +0.2 |
| ViDoRe_tabfquad | 94.7 | 97.0 | 96.7 | +0.3 |
| ViDoRe_tatdqa | 59.7 | 70.0 | 70.0 | -0.0 |
| ViDoSeek-doc | 84.7 | 86.2 | 86.1 | +0.1 |
| ViDoSeek-page-fixed | nan | 87.0 | 88.4 | -1.4 |
| VisRAG_ArxivQA | 78.4 | 88.0 | 88.2 | -0.2 |
| VisRAG_ChartQA | 87.2 | 88.9 | 88.3 | +0.6 |
| VisRAG_InfoVQA | 90.5 | 94.7 | 94.7 | -0.1 |
| VisRAG_MP-DocVQA | 80.5 | 88.8 | 89.3 | -0.5 |
| VisRAG_PlotQA | 66.0 | 73.9 | 74.8 | -0.9 |
| VisRAG_SlideVQA | 93.1 | 96.8 | 96.7 | +0.1 |
