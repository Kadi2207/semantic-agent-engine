"""Controle automatique de fidelite du RAG par NLI local (exploratoire). Aucun appel Groq.

PROTOCOLE FIGE AVANT TOUTE EXECUTION :
- modele : MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (revision enregistree dans le JSON) ;
- prediction = etiquette de score maximal parmi entailment, neutral, contradiction ; aucun seuil ;
- unite d'analyse : chaque phrase d'une synthese RAG en cache (dev : rag_dev.json et data/processed/groq_rag_cache.jsonl ;
  test2 : test2.json et data/processed/groq_rag_test2_cache.jsonl, tous les projets dont la synthese est en cache) ;
- premisse = texte integral des unites citees par la phrase (enonce puis preuve, dans l'ordre des citations) ;
  hypothese = la phrase, sans ses citations ;
- les reponses qui ne sont pas des syntheses acceptees (rejetees par le code, abstentions, extractives, conflits) sont
  exclues et comptees a part.

    uv run python eval/nli_faithfulness.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
LABELS = ["entailment", "neutral", "contradiction"]
OUT_PATH = ROOT / "eval" / "results" / "rag_nli.json"
SPLITS = {
    "dev": {"results": "eval/results/rag_dev.json", "cache": "data/processed/groq_rag_cache.jsonl", "data": "data/raw"},
    "test2": {"results": "eval/results/test2.json", "cache": "data/processed/groq_rag_test2_cache.jsonl", "data": "data/test2"},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sentences_of(split: str) -> tuple[list[dict[str, object]], Counter]:
    """Phrases des syntheses acceptees en cache, avec la premisse construite depuis les unites citees."""
    from src.agent import rag as rag_module
    from src.ingest.groq_client import CachedGroq
    from src.memory.corpus import load_corpus

    spec = SPLITS[split]
    units = {c.unit.unit_id: c.unit for c in load_corpus(ROOT / spec["data"])}
    data = json.loads((ROOT / spec["results"]).read_text(encoding="utf-8"))
    rows = data["results"]["per_query"] if split == "dev" else data["stages"]["rag"]["per_query"]
    client = CachedGroq(model=rag_module.RAG_MODEL, temperature=0.0, cache_path=ROOT / spec["cache"], min_seconds_between_calls=0,
                        json_mode=True, cache_only=True, max_completion_tokens=rag_module.RAG_MAX_COMPLETION_TOKENS,
                        extra_body=rag_module.RAG_EXTRA)
    excluded: Counter = Counter()
    out = []
    for row in rows:
        if row["status"] != "synthese":
            excluded[row["status"]] += 1
            continue
        provided = list(row["provided_ids"])
        user = rag_module.build_user_prompt(str(row["query"]), [units[i] for i in provided])
        raw = client._cache.get(client.key(rag_module.RAG_SYSTEM_PROMPT, user))  # noqa: SLF001
        if raw is None:
            excluded["synthese_absente_du_cache"] += 1
            continue
        _, sentences, rejection = rag_module.check_synthesis(raw, provided)
        if rejection:
            excluded["rejetee"] += 1
            continue
        for s in sentences:
            premise = " ".join(f"{units[i].statement} {units[i].evidence}" for i in s.unit_ids)
            out.append({"query_id": row["query_id"], "piege": row["piege"], "hypothesis": s.text, "cited_ids": list(s.unit_ids),
                        "premise": premise})
    return out, excluded


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]))
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    assert sorted(id2label.values()) == sorted(LABELS), id2label

    payload: dict[str, object] = {
        "description": "Controle automatique de fidelite du RAG par NLI local",
        "warning": "exploratoire : un modele NLI est imparfait, surtout en francais et sur des phrases de synthese ; ce controle "
                   "complete le controle des citations et ne remplace pas une evaluation humaine, non realisee",
        "date": date.today().isoformat(),
        "script": "eval/nli_faithfulness.py",
        "protocol": {"model": MODEL_ID, "revision": path.name, "prediction": "etiquette de score maximal, aucun seuil",
                     "premise": "texte integral des unites citees par la phrase (enonce puis preuve)",
                     "hypothesis": "la phrase de la synthese, sans ses citations",
                     "excluded": "reponses qui ne sont pas des syntheses acceptees (rejetees, abstentions, extractives, conflits)",
                     "fixed_before_execution": True},
        "sources_sha256": {p: sha(ROOT / p) for spec in SPLITS.values() for p in (spec["results"], spec["cache"]) if (ROOT / p).exists()},
        "splits": {},
    }
    for split in SPLITS:
        items, excluded = sentences_of(split)
        for item in items:
            enc = tokenizer(item["premise"], item["hypothesis"], truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits[0], dim=-1).tolist()
            item["label"] = id2label[max(range(len(probs)), key=probs.__getitem__)]
            item["scores"] = {id2label[i]: round(p, 4) for i, p in enumerate(probs)}
        counts = Counter(i["label"] for i in items)
        n = len(items)
        payload["splits"][split] = {  # type: ignore[index]
            "sentences": n,
            "syntheses": len({i["query_id"] for i in items}),
            "excluded_responses": dict(excluded),
            "counts": {lab: counts.get(lab, 0) for lab in LABELS},
            "rates": {lab: round(counts.get(lab, 0) / n, 4) if n else 0.0 for lab in LABELS},
            "not_entailed": [i for i in items if i["label"] != "entailment"],
        }
        print(split, n, payload["splits"][split]["counts"], payload["splits"][split]["rates"], dict(excluded))  # type: ignore[index]
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultats ecrits dans {OUT_PATH}")


if __name__ == "__main__":
    main()
