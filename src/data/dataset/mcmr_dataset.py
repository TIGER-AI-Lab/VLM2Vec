import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_QUERY_INSTRUCTION = "Find an image-text pair that matches the given query"
DEFAULT_QUERY_FILENAME = "query.jsonl"
DEFAULT_CANDIDATE_FILENAME = "candidates.jsonl"
DEFAULT_IMAGE_DIRNAME = "images"

# Candidate text is concatenated from these fields.
CANDIDATE_TEXT_FIELDS: Tuple[str, ...] = (
    "title",
    "description",
    "features",
    "price",
    "Date First Available",
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_mcmr_paths(
    data_path: Optional[str] = None,
    query_file: Optional[str] = None,
    candidate_file: Optional[str] = None,
    image_root: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if data_path:
        if query_file is None:
            query_file = os.path.join(data_path, DEFAULT_QUERY_FILENAME)
        elif not os.path.isabs(query_file):
            query_file = os.path.join(data_path, query_file)

        if candidate_file is None:
            candidate_file = os.path.join(data_path, DEFAULT_CANDIDATE_FILENAME)
        elif not os.path.isabs(candidate_file):
            candidate_file = os.path.join(data_path, candidate_file)

        if image_root is None:
            image_root = os.path.join(data_path, DEFAULT_IMAGE_DIRNAME)
        elif not os.path.isabs(image_root):
            image_root = os.path.join(data_path, image_root)

    return query_file, candidate_file, image_root


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_stringify(v) for v in value]
        parts = [p for p in parts if p]
        return " ".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def build_mcmr_query_text(query: Any, query_instruction: str = DEFAULT_QUERY_INSTRUCTION) -> str:
    query_text = _stringify(query)
    inst_text = _stringify(query_instruction)
    if inst_text and query_text:
        return f"{inst_text} {query_text}"
    return query_text or inst_text


def build_mcmr_candidate_text(
    row: Dict[str, Any],
    text_fields: Sequence[str] = CANDIDATE_TEXT_FIELDS,
) -> str:
    parts: List[str] = []
    for field in text_fields:
        value = _stringify(row.get(field))
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts).strip()


def _image_name_from_url(image_url: str) -> str:
    if not image_url:
        return ""
    image_url = image_url.strip()
    if not image_url:
        return ""
    image_url = image_url.split("?", 1)[0]
    return os.path.basename(image_url)


def resolve_candidate_image_path(row: Dict[str, Any], image_root: str) -> Optional[str]:
    images = row.get("images", [])
    if isinstance(images, dict):
        images = [images]

    for image_item in images:
        image_name = ""
        if isinstance(image_item, str):
            image_name = _image_name_from_url(image_item)
        elif isinstance(image_item, dict):
            image_name = _image_name_from_url(
                image_item.get("url")
                or image_item.get("path")
                or image_item.get("image")
                or ""
            )
        if not image_name:
            continue
        image_path = os.path.join(image_root, image_name)
        if os.path.isfile(image_path):
            return image_path
    return None


def load_mcmr_query_rows(
    query_file: str,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
) -> List[Dict[str, Any]]:
    rows = read_jsonl(query_file)
    parsed: List[Dict[str, Any]] = []

    for row in rows:
        qid = str(row.get("qid", "")).strip()
        query_text = build_mcmr_query_text(row.get("query", ""), query_instruction=query_instruction)
        pos_ids = row.get("pos_ids", [])

        if isinstance(pos_ids, str):
            try:
                pos_ids = json.loads(pos_ids)
            except json.JSONDecodeError:
                pos_ids = [pos_ids]
        elif not isinstance(pos_ids, list):
            pos_ids = [pos_ids] if pos_ids else []

        labels = [str(x) for x in pos_ids if str(x).strip()]
        if not qid or not query_text:
            continue

        parsed.append({
            "qry_id": qid,
            "qry_text": query_text,
            "label_names": labels,
        })
    return parsed


def load_mcmr_candidate_rows(candidate_file: str, image_root: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(candidate_file)
    parsed: List[Dict[str, Any]] = []

    for row in rows:
        cand_id = str(row.get("candidate_id", "")).strip()
        cand_text = build_mcmr_candidate_text(row)
        image_path = resolve_candidate_image_path(row, image_root=image_root)

        if not cand_id:
            continue
        if image_path is None:
            raise FileNotFoundError(
                f"Cannot find local image for candidate_id={cand_id} under image_root={image_root}"
            )
        if not cand_text:
            cand_text = f"candidate_id: {cand_id}"

        parsed.append({
            "cand_id": cand_id,
            "cand_text": cand_text,
            "cand_image_path": image_path,
        })

    return parsed
