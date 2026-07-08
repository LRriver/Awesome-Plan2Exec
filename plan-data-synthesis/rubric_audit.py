"""Rubric audit report for strategy-iteration runs.

This module is intentionally outside the production extraction path. It reads
evaluated candidates and emits aggregate signals for revising rubrics, negative
types, budgets, and model routing.
"""
import json
from collections import Counter, defaultdict
from statistics import mean

import config


def _score_bucket(score: float) -> str:
    buckets = getattr(config, "QUALITY_BUCKETS", {"strong": 8.5, "medium": 6.0, "weak": 3.0})
    if score >= buckets.get("strong", 8.5):
        return "strong"
    if score >= buckets.get("medium", 6.0):
        return "medium"
    return "weak"


def _summarize_scores(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0, "avg_score": None, "min_score": None, "max_score": None}
    return {
        "count": len(scores),
        "avg_score": round(mean(scores), 4),
        "min_score": min(scores),
        "max_score": max(scores),
    }


def _candidate_type(plan: dict) -> str:
    return plan.get("negative_type") or "positive"


def build_audit_report(records: list[dict]) -> dict:
    """Build aggregate rubric and distribution statistics."""
    all_scores = []
    scores_by_difficulty = defaultdict(list)
    scores_by_negative_type = defaultdict(list)
    bucket_by_difficulty = defaultdict(Counter)
    evaluation_sources = Counter()
    criteria_total = Counter()
    criteria_failures = Counter()
    criteria_scores = defaultdict(list)

    for record in records:
        difficulty = record.get("difficulty", "unknown")
        for item in record.get("evaluated_plans", []):
            evaluation = item.get("evaluation", {})
            score = evaluation.get("final_score")
            if not isinstance(score, int | float):
                continue

            score = float(score)
            plan = item.get("plan", {})
            negative_type = _candidate_type(plan)
            all_scores.append(score)
            scores_by_difficulty[difficulty].append(score)
            scores_by_negative_type[negative_type].append(score)
            bucket_by_difficulty[difficulty][_score_bucket(score)] += 1
            evaluation_sources[evaluation.get("evaluation_source", "unknown")] += 1

            for criterion in evaluation.get("rubric_criteria", []) or []:
                criterion_id = criterion.get("criterion_id")
                criterion_score = criterion.get("score")
                if not criterion_id or not isinstance(criterion_score, int | float):
                    continue
                criterion_score = float(criterion_score)
                criteria_total[criterion_id] += 1
                criteria_scores[criterion_id].append(criterion_score)
                if criterion_score < 0.7:
                    criteria_failures[criterion_id] += 1

    return {
        "mode": "iteration",
        "strategy_checkpoint": getattr(config, "STRATEGY_CHECKPOINT", "controlled_v1"),
        "overall": {
            **_summarize_scores(all_scores),
            "candidate_count": len(all_scores),
        },
        "by_difficulty": {
            difficulty: {
                **_summarize_scores(scores),
                "quality_buckets": dict(bucket_by_difficulty[difficulty]),
            }
            for difficulty, scores in sorted(scores_by_difficulty.items())
        },
        "by_negative_type": {
            negative_type: _summarize_scores(scores)
            for negative_type, scores in sorted(scores_by_negative_type.items())
        },
        "by_evaluation_source": dict(evaluation_sources),
        "rubric_criteria": {
            criterion_id: {
                "count": criteria_total[criterion_id],
                "avg_score": round(mean(criteria_scores[criterion_id]), 4),
                "failure_count": criteria_failures[criterion_id],
                "failure_rate": round(criteria_failures[criterion_id] / criteria_total[criterion_id], 4),
            }
            for criterion_id in sorted(criteria_total)
        },
    }


def load_evaluated_records() -> list[dict]:
    """Load evaluated plan records from the configured path."""
    if not config.EVALUATED_PLANS_FILE.exists():
        return []
    records = []
    with open(config.EVALUATED_PLANS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_audit_report(records: list[dict]) -> dict:
    """Write the rubric audit report and return it."""
    report = build_audit_report(records)
    output_path = getattr(config, "RUBRIC_AUDIT_REPORT_FILE", config.OUTPUT_DIR / "rubric_audit_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> dict:
    """Generate the audit report from evaluated plans."""
    records = load_evaluated_records()
    report = write_audit_report(records)
    print(f"[INFO] Rubric audit report written: {config.RUBRIC_AUDIT_REPORT_FILE}")
    return report
