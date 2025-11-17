# (repo, subset, split)
import os

BASE_RAW_DATA_DIR = ""

EVAL_DATASET_HF_PATH = {
    # Video-RET
    "MSR-VTT": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "MSR-VTT"), "test_1k", "test"),
    "MSVD": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "MSVD"), None, "test"),
    "DiDeMo": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "DiDeMo"), None, "test"),
    "YouCook2": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "YouCook2"), None, "val"),
    "VATEX": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "VATEX"), None, "test"),

    # Video-CLS
    "HMDB51": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "HMDB51"), None, "test"),
    "UCF101": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "UCF101"), None, "test"),
    "Breakfast": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "Breakfast"), None, "test"),
    "Kinetics-700": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "Kinetics-700"), None, "test"),
    "SmthSmthV2": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "SmthSmthV2"), None, "test"),

    # Video-MRET
    "QVHighlight": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "QVHighlight"), None, "test"),
    "Charades-STA": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "Charades-STA"), None, "test"),
    "MomentSeeker": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "MomentSeeker"), None, "test"),
    "MomentSeeker_1k8": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "MomentSeeker_1k8"), None, "test"),

    # Video-QA
    "NExTQA": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "NExTQA"), "MC", "test"),
    "EgoSchema": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "EgoSchema"), "Subset", "test"),
    "MVBench": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "MVBench"), None, "train"),
    "Video-MME": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "Video-MME"), None, "test"),
    "ActivityNetQA": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "ActivityNetQA"), None, "test"),

    # Visdoc-ViDoRe
    "ViDoRe_arxivqa": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "arxivqa_test_subsampled_beir"), None, "test"),
    "ViDoRe_docvqa": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "docvqa_test_subsampled_beir"), None, "test"),
    "ViDoRe_infovqa": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "infovqa_test_subsampled_beir"), None, "test"),
    "ViDoRe_tabfquad": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "tabfquad_test_subsampled_beir"), None, "test"),
    "ViDoRe_tatdqa": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "tatdqa_test_beir"), None, "test"),
    "ViDoRe_shiftproject": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "shiftproject_test_beir"), None, "test"),
    "ViDoRe_syntheticDocQA_artificial_intelligence": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "syntheticDocQA_artificial_intelligence_test_beir"), None, "test"),
    "ViDoRe_syntheticDocQA_energy": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "syntheticDocQA_energy_test_beir"), None, "test"),
    "ViDoRe_syntheticDocQA_government_reports": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "syntheticDocQA_government_reports_test_beir"), None, "test"),
    "ViDoRe_syntheticDocQA_healthcare_industry": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "syntheticDocQA_healthcare_industry_test_beir"), None, "test"),

    # Visdoc-VisRAG
    "VisRAG_ArxivQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "VisRAG-Ret-Test-ArxivQA"), None, "train"),
    "VisRAG_ChartQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "VisRAG-Ret-Test-ChartQA"), None, "train"),
    "VisRAG_MP-DocVQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "VisRAG-Ret-Test-MP-DocVQA"), None, "train"),
    "VisRAG_SlideVQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "VisRAG-Ret-Test-SlideVQA"), None, "train"),
    "VisRAG_InfoVQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "VisRAG-Ret-Test-InfoVQA"), None, "train"),
    "VisRAG_PlotQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "VisRAG-Ret-Test-PlotQA"), None, "train"),

    # Visdoc-ViDoSeek
    "ViDoSeek-doc": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "ViDoSeek"), None, "test"),
    "ViDoSeek-page": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "ViDoSeek-page"), None, "test"),
    "MMLongBench-doc": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "MMLongBench-doc"), None, "test"),
    "MMLongBench-page": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "MMLongBench"), None, "test"),

    # Visdoc-ViDoRe_v2
    "ViDoRe_esg_reports_human_labeled_v2": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "esg_reports_human_labeled_v2"), None, "test"),
    "ViDoRe_biomedical_lectures_v2": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "biomedical_lectures_v2"), "english", "test"),
    "ViDoRe_biomedical_lectures_v2_multilingual": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "biomedical_lectures_v2"), None, "test"),
    "ViDoRe_economics_reports_v2": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "economics_reports_v2"), "english", "test"),
    "ViDoRe_economics_reports_v2_multilingual": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "economics_reports_v2"), None, "test"),
    "ViDoRe_esg_reports_v2": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "esg_reports_v2"), "english", "test"),
    "ViDoRe_esg_reports_v2_multilingual": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "esg_reports_v2"), None, "test"),

    # Image-CLS
    "ImageNet-1K": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "ImageNet-1K", "test"),
    "N24News": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "N24News", "test"),
    "HatefulMemes": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "HatefulMemes", "test"),
    "VOC2007": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "VOC2007", "test"),
    "SUN397": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "SUN397", "test"),
    "Place365": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "Place365", "test"),
    "ImageNet-A": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "ImageNet-A", "test"),
    "ImageNet-R": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "ImageNet-R", "test"),
    "ObjectNet": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "ObjectNet", "test"),
    "Country211": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "Country211", "test"),
    # Image-QA
    "OK-VQA": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "OK-VQA", "test"),
    "A-OKVQA": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "A-OKVQA", "test"),
    "DocVQA": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "DocVQA", "test"),
    "InfographicsVQA": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "InfographicsVQA", "test"),
    "ChartQA": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "ChartQA", "test"),
    "Visual7W": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "Visual7W", "test"),
    "ScienceQA": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "ScienceQA", "test"),
    "VizWiz": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "VizWiz", "test"),
    "GQA": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "GQA", "test"),
    "TextVQA": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "TextVQA", "test"),
    # Image-RET
    "VisDial": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "VisDial", "test"),
    "CIRR": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "CIRR", "test"),
    "VisualNews_t2i": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "VisualNews_t2i", "test"),
    "VisualNews_i2t": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "VisualNews_i2t", "test"),
    "MSCOCO_t2i": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "MSCOCO_t2i", "test"),
    "MSCOCO_i2t": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "MSCOCO_i2t", "test"),
    "NIGHTS": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "NIGHTS", "test"),
    "WebQA": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "WebQA", "test"),
    "FashionIQ": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "FashionIQ", "test"),
    "Wiki-SS-NQ": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "Wiki-SS-NQ", "test"),
    "OVEN": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "OVEN", "test"),
    "EDIS": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "EDIS", "test"),
    # Image-VG
    "MSCOCO": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "MSCOCO", "test"),
    "RefCOCO": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "RefCOCO", "test"),
    "RefCOCO-Matching": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "RefCOCO-Matching", "test"),
    "Visual7W-Pointing": (os.path.join(BASE_RAW_DATA_DIR, "image-tasks"), "Visual7W-Pointing", "test"),
}
