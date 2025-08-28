# (repo, subset, split)
EVAL_DATASET_HF_PATH = {
    # Video-RET
    "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
    "MSVD": ("VLM2Vec/MSVD", None, "test"),
    "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
    # "YouCook2": ("VLM2Vec/YouCook2", None, "val"), # HF version compatibility issue
    "YouCook2": ("lmms-lab/YouCook2", None, "val"),
    "VATEX": ("VLM2Vec/VATEX", None, "test"),

    # Video-CLS
    "HMDB51": ("VLM2Vec/HMDB51", None, "test"),
    "UCF101": ("VLM2Vec/UCF101", None, "test"),
    "Breakfast": ("VLM2Vec/Breakfast", None, "test"),
    "Kinetics-700": ("VLM2Vec/Kinetics-700", None, "test"),
    "SmthSmthV2": ("VLM2Vec/SmthSmthV2", None, "test"),

    # Video-MRET
    "QVHighlight": ("VLM2Vec/QVHighlight", None, "test"),
    "Charades-STA": ("VLM2Vec/Charades-STA", None, "test"),
    "MomentSeeker": ("VLM2Vec/MomentSeeker", None, "test"),
    "MomentSeeker_1k8": ("VLM2Vec/MomentSeeker_1k8", None, "test"),

    # Video-QA
    "NExTQA": ("VLM2Vec/NExTQA", "MC", "test"),
    "EgoSchema": ("VLM2Vec/EgoSchema", "Subset", "test"),
    "MVBench": ("VLM2Vec/MVBench", None, "train"),
    "Video-MME": ("VLM2Vec/Video-MME", None, "test"),
    "ActivityNetQA": ("VLM2Vec/ActivityNetQA", None, "test"),

    # Visdoc-ViDoRe
    "ViDoRe_arxivqa": ("vidore/arxivqa_test_subsampled_beir", None, "test"),
    "ViDoRe_docvqa": ("vidore/docvqa_test_subsampled_beir", None, "test"),
    "ViDoRe_infovqa": ("vidore/infovqa_test_subsampled_beir", None, "test"),
    "ViDoRe_tabfquad": ("vidore/tabfquad_test_subsampled_beir", None, "test"),
    "ViDoRe_tatdqa": ("vidore/tatdqa_test_beir", None, "test"),
    "ViDoRe_shiftproject": ("vidore/shiftproject_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_artificial_intelligence": ("vidore/syntheticDocQA_artificial_intelligence_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_energy": ("vidore/syntheticDocQA_energy_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_government_reports": ("vidore/syntheticDocQA_government_reports_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_healthcare_industry": ("vidore/syntheticDocQA_healthcare_industry_test_beir", None, "test"),

    # Visdoc-VisRAG
    "VisRAG_ArxivQA": ("openbmb/VisRAG-Ret-Test-ArxivQA", None, "train"),
    "VisRAG_ChartQA": ("openbmb/VisRAG-Ret-Test-ChartQA", None, "train"),
    "VisRAG_MP-DocVQA": ("openbmb/VisRAG-Ret-Test-MP-DocVQA", None, "train"),
    "VisRAG_SlideVQA": ("openbmb/VisRAG-Ret-Test-SlideVQA", None, "train"),
    "VisRAG_InfoVQA": ("openbmb/VisRAG-Ret-Test-InfoVQA", None, "train"),
    "VisRAG_PlotQA": ("openbmb/VisRAG-Ret-Test-PlotQA", None, "train"),

    # Visdoc-ViDoSeek
    "ViDoSeek-doc": ("VLM2Vec/ViDoSeek", None, "test"),
    "ViDoSeek-page": ("VLM2Vec/ViDoSeek-page", None, "test"),
    "MMLongBench-doc": ("VLM2Vec/MMLongBench-doc", None, "test"),
    "MMLongBench-page": ("VLM2Vec/MMLongBench", None, "test"),

    # Visdoc-ViDoRe_v2
    "ViDoRe_esg_reports_human_labeled_v2": ("vidore/esg_reports_human_labeled_v2", None, "test"),
    "ViDoRe_biomedical_lectures_v2": ("vidore/biomedical_lectures_v2", "english", "test"),
    "ViDoRe_biomedical_lectures_v2_multilingual": ("vidore/biomedical_lectures_v2", None, "test"),
    "ViDoRe_economics_reports_v2": ("vidore/economics_reports_v2", "english", "test"),
    "ViDoRe_economics_reports_v2_multilingual": ("vidore/economics_reports_v2", None, "test"),
    "ViDoRe_esg_reports_v2": ("vidore/esg_reports_v2", "english", "test"),
    "ViDoRe_esg_reports_v2_multilingual": ("vidore/esg_reports_v2", None, "test"),
}
