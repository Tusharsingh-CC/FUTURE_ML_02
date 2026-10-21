"""Triage new tickets with the trained model.

Examples:
    python -m src.predict --text "URGENT!! our site is down and customers cannot pay"
    python -m src.predict --input new_tickets.csv --text-col "Ticket Description" --output triaged.csv
"""
import argparse

import joblib
import pandas as pd

from src.preprocess import clean_text
from src.priority import RANK, hybrid_priority, score_ticket


def triage(texts, bundle):
    """Return a DataFrame sorted so the most urgent tickets come first."""
    cat_model, prio_model = bundle["category_model"], bundle["priority_model"]
    thr = bundle["meta"]["review_threshold"]
    clean = [clean_text(t) for t in texts]
    proba = cat_model.predict_proba(clean)
    cats = cat_model.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    ml_prio = prio_model.predict(clean) if prio_model is not None else [None] * len(texts)

    rows = []
    for raw, cat, c, mlp in zip(texts, cats, conf, ml_prio):
        rule = score_ticket(raw, cat)
        final = hybrid_priority(mlp, rule)
        rows.append({
            "ticket_text": raw, "predicted_category": cat, "category_confidence": round(float(c), 3),
            "predicted_priority": final, "ml_priority": mlp, "rule_score": rule.score,
            "priority_reasons": " | ".join(rule.reasons) or "no urgency signals found",
            "needs_human_review": bool(c < thr),
        })
    out = pd.DataFrame(rows)
    out["_rank"] = out["predicted_priority"].map(RANK).fillna(-1)
    return out.sort_values(["_rank", "rule_score"], ascending=False).drop(columns="_rank").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ticket_triage.joblib")
    ap.add_argument("--text", nargs="+", help="one or more ticket texts")
    ap.add_argument("--input", help="CSV file with tickets")
    ap.add_argument("--text-col", default="Ticket Description")
    ap.add_argument("--output", help="where to save the triaged CSV")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    if a.text:
        texts = a.text
    elif a.input:
        texts = pd.read_csv(a.input)[a.text_col].fillna("").astype(str).tolist()
    else:
        ap.error("provide --text or --input")

    res = triage(texts, bundle)
    if a.output:
        res.to_csv(a.output, index=False)
        print(f"Saved {len(res)} triaged tickets to {a.output}")
    pd.set_option("display.max_colwidth", 70, "display.width", 200)
    print(res[["ticket_text", "predicted_category", "category_confidence", "predicted_priority",
               "needs_human_review"]].head(20).to_string(index=False))
    if a.text:
        for _, r in res.iterrows():
            print(f"\nWhy '{r['predicted_priority']}': {r['priority_reasons']}")


if __name__ == "__main__":
    main()
