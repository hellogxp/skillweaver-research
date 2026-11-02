"""Oracle step-count and alternative-encoder controls."""

from __future__ import annotations

from experiments.config import ROOT, get_options, seed_everything
OPTIONS = get_options(parse_cli=__name__ == "__main__")
seed_everything(OPTIONS.seed)
DATA_DIR = OPTIONS.data_dir.resolve()
RESULTS_DIR = OUTPUT_DIR = OPTIONS.output_dir.resolve()

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict


from skillweaver.core.decomposer import LocalDecomposer, parse_subtasks, SYSTEM_PROMPT
from skillweaver.core.models import Skill
from skillweaver.core.retriever import SkillRetriever
from skillweaver.core.pipeline import build_hint_set

MODEL_7B = OPTIONS.model
ENCODER_MINILM = OPTIONS.encoder
SKILLS_PATH = str(DATA_DIR / "processed_v3" / "skill_pool.jsonl")
QUERIES_PATH = OPTIONS.query_file or (str(DATA_DIR / "benchmark_v3" / "compositional_queries.jsonl"))

H = 15
TOP_K = 10


# ---------- Constrained prompt (oracle K) ----------
CONSTRAINED_USER_TEMPLATE = """Decompose this query into EXACTLY {k} atomic sub-tasks. Output a JSON array of EXACTLY {k} strings ONLY.

Query: {query}

JSON array of {k} strings:"""


def constrained_decompose(decomposer: LocalDecomposer, query: str, k: int):
    """Decompose with oracle step-count constraint."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": CONSTRAINED_USER_TEMPLATE.format(query=query, k=k)},
    ]
    text = decomposer._generate(messages)
    return parse_subtasks(text, query)


# ---------- Metrics ----------
def compute_metrics(subtasks, candidates, query_obj, skills_by_id):
    """Returns dict with da, da_pm1, cat_r1, cat_r10."""
    gt_subs = query_obj["subtasks"]
    gt_steps = len(gt_subs)
    gt_cats = [s["required_category"] for s in gt_subs]

    pred_steps = len(subtasks)
    da = 1.0 if pred_steps == gt_steps else 0.0
    da_pm1 = 1.0 if abs(pred_steps - gt_steps) <= 1 else 0.0

    # Step-level recall (aligned by index, up to gt_steps)
    n_eval = gt_steps
    cat_r1_hits = 0
    cat_r10_hits = 0
    for i in range(n_eval):
        gt_cat = gt_cats[i]
        if gt_cat is None or i >= len(candidates) or not candidates[i]:
            continue
        cands = candidates[i]
        # CatR@1
        if gt_cat in (cands[0].skill.categories or []):
            cat_r1_hits += 1
        # CatR@10
        for match in cands[:10]:
            if gt_cat in (match.skill.categories or []):
                cat_r10_hits += 1
                break
    cat_r1 = cat_r1_hits / n_eval if n_eval else 0.0
    cat_r10 = cat_r10_hits / n_eval if n_eval else 0.0
    return {
        "da": da,
        "da_pm1": da_pm1,
        "cat_r1": cat_r1,
        "cat_r10": cat_r10,
        "pred_steps": pred_steps,
        "gt_steps": gt_steps,
    }


def aggregate(metrics_list):
    if not metrics_list:
        return {}
    n = len(metrics_list)
    return {
        "n": n,
        "da": sum(m["da"] for m in metrics_list) / n,
        "da_pm1": sum(m["da_pm1"] for m in metrics_list) / n,
        "cat_r1": sum(m["cat_r1"] for m in metrics_list) / n,
        "cat_r10": sum(m["cat_r10"] for m in metrics_list) / n,
        "avg_pred_steps": sum(m["pred_steps"] for m in metrics_list) / n,
        "avg_gt_steps": sum(m["gt_steps"] for m in metrics_list) / n,
    }


# ---------- B1 ----------
def run_b1_constrained_vanilla(decomposer, retriever, queries, skills_by_id):
    """Vanilla with oracle step-count constraint, 300 queries."""
    print("\n" + "=" * 60)
    print("B1: Step-count-constrained vanilla (oracle K)")
    print("=" * 60)
    results = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        query_text = q["query"]
        gt_k = len(q["subtasks"])
        subtasks = constrained_decompose(decomposer, query_text, gt_k)
        texts = [st.description for st in subtasks]
        cands = retriever.search_batch(texts)
        m = compute_metrics(subtasks, cands, q, skills_by_id)
        results.append(m)
        if (qi + 1) % 50 == 0:
            agg = aggregate(results)
            print(f"  [{qi+1}/{len(queries)}] DA={agg['da']:.3f} CatR@1={agg['cat_r1']:.3f}", flush=True)
    elapsed = time.time() - t0
    summary = aggregate(results)
    summary["elapsed_sec"] = elapsed
    summary["per_query"] = results
    out_path = RESULTS_DIR / "constrained_vanilla_oracle_k.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved -> {out_path}")
    print(f"  Final: DA={summary['da']:.3f} DA±1={summary['da_pm1']:.3f} CatR@1={summary['cat_r1']:.3f} CatR@10={summary['cat_r10']:.3f}")
    return summary


# ---------- B2 ----------
def try_load_bge_encoder():
    """Use only an explicitly selected alternative; do not probe workstation paths."""
    return OPTIONS.alternative_encoder


def run_b2_stronger_encoder(decomposer, queries, skills, skills_by_id, encoder_path):
    """SAD with a stronger encoder, 50 queries."""
    print("\n" + "=" * 60)
    print(f"B2: Stronger encoder = {encoder_path}")
    print("=" * 60)
    retriever = SkillRetriever(encoder_name=encoder_path, use_body=False, top_k=TOP_K)
    retriever.build_index(skills)

    sub_queries = queries[:50]
    results = {"vanilla": [], "sad": []}
    for qi, q in enumerate(sub_queries):
        query_text = q["query"]
        subtasks_v = decomposer.decompose(query_text)
        texts_v = [st.description for st in subtasks_v]
        cands_v = retriever.search_batch(texts_v)
        results["vanilla"].append(compute_metrics(subtasks_v, cands_v, q, skills_by_id))

        hints = build_hint_set(cands_v, H)
        subtasks_s = decomposer.decompose_with_hints(query_text, hints)
        texts_s = [st.description for st in subtasks_s]
        cands_s = retriever.search_batch(texts_s)
        results["sad"].append(compute_metrics(subtasks_s, cands_s, q, skills_by_id))

        if (qi + 1) % 10 == 0:
            print(f"  [{qi+1}/{len(sub_queries)}] vanilla DA={aggregate(results['vanilla'])['da']:.3f}", flush=True)

    summary = {
        "encoder": encoder_path,
        "vanilla": aggregate(results["vanilla"]),
        "sad": aggregate(results["sad"]),
    }
    out_path = RESULTS_DIR / "encoder_swap_50q.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved -> {out_path}")
    return summary


# ---------- Main ----------
def main():
    # Load skills
    skills = []
    skills_by_id = {}
    for line in open(SKILLS_PATH):
        d = json.loads(line)
        sk = Skill.from_dict(d) if hasattr(Skill, "from_dict") else Skill(**d)
        skills.append(sk)
        skills_by_id[sk.skill_id] = sk
    print(f"Loaded {len(skills)} skills")

    queries = [json.loads(line) for line in open(QUERIES_PATH)]
    print(f"Loaded {len(queries)} queries")

    # Build MiniLM retriever for B1
    print("Building MiniLM retriever...")
    retriever = SkillRetriever(encoder_name=ENCODER_MINILM, use_body=False, top_k=TOP_K)
    retriever.build_index(skills)

    # Load 7B decomposer
    print("Loading Qwen2.5-7B-Instruct...")
    decomposer = LocalDecomposer(model_path=MODEL_7B, temperature=0.1, seed=OPTIONS.seed)

    # B1
    b1 = run_b1_constrained_vanilla(decomposer, retriever, queries, skills_by_id)

    # B2 — only if a stronger encoder is available locally
    bge_path = try_load_bge_encoder()
    if bge_path:
        try:
            b2 = run_b2_stronger_encoder(decomposer, queries, skills, skills_by_id, bge_path)
        except Exception as e:
            print(f"  B2 failed: {e}")
            b2 = {"error": str(e)}
    else:
        print("\nB2: no BGE/MPNet/E5 encoder found locally — SKIPPED")
        b2 = {"skipped": "no stronger encoder available locally"}

    # Combined summary
    combined = {"b1_constrained_vanilla": b1, "b2_stronger_encoder": b2}
    summary_path = RESULTS_DIR / "step_count_controls_summary.json"
    summary_path.write_text(json.dumps(combined, indent=2, default=str))
    print(f"\nAll done. Summary saved -> {summary_path}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
