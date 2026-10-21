"""Train, compare and evaluate the ticket classifier (+ priority model if labels exist).

Usage:
    python -m src.train --data data/sample_tickets.csv
    python -m src.train --data path/to/your.csv --text-cols "Subject" "Body" --category-col "Queue"
"""
import argparse
import json
import os
import time
import warnings
from collections import Counter

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from src.load_data import load_dataset  # noqa: E402
from src.preprocess import clean_text  # noqa: E402
from src.priority import ORDER, hybrid_priority, labels_compatible, score_ticket  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42


def tfidf():
    # unigrams + bigrams capture phrases such as "not working" or "charged twice"
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)


def make_models():
    return {
        "Logistic Regression": Pipeline([("tfidf", tfidf()),
                                         ("clf", LogisticRegression(max_iter=1000, C=5, class_weight="balanced"))]),
        "Linear SVM": Pipeline([("tfidf", tfidf()),
                                ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3))]),
        "Complement Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", ComplementNB(alpha=0.3))]),
        "Random Forest": Pipeline([("tfidf", tfidf()),
                                   ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                                  class_weight="balanced_subsample",
                                                                  random_state=SEED))]),
    }


def metrics_dict(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def plot_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 3, 1.0 * len(labels) + 2.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return cm


def plot_model_comparison(comp, path):
    names = list(comp)
    cv = [comp[n]["cv_macro_f1_mean"] for n in names]
    te = [comp[n]["test"]["macro_f1"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, cv, 0.4, label="Cross-validation macro-F1")
    ax.bar(x + 0.2, te, 0.4, label="Held-out test macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def top_terms(X_train, y_train, n=8):
    """Most indicative words per category (from a logistic-regression fit) - for explainability."""
    p = Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=1000, C=5))]).fit(X_train, y_train)
    names = np.array(p.named_steps["tfidf"].get_feature_names_out())
    coef = p.named_steps["clf"].coef_
    classes = p.named_steps["clf"].classes_
    if len(classes) == 2:  # binary case: sklearn stores one row
        coef = np.vstack([-coef[0], coef[0]])
    return {c: list(names[np.argsort(coef[i])[::-1][:n]]) for i, c in enumerate(classes)}


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-cols", nargs="+")
    ap.add_argument("--category-col")
    ap.add_argument("--priority-col")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--review-threshold", type=float, default=0.5,
                    help="tickets with category confidence below this are flagged for human review")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--model-dir", default="models")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.model_dir, exist_ok=True)

    # ---------------- 1. load + clean ----------------
    df, meta = load_dataset(a.data, a.text_cols, a.category_col, a.priority_col)
    n0 = len(df)
    df["clean"] = df["raw_text"].map(clean_text)
    df = df[df["clean"].str.len() > 0].drop_duplicates(subset=["clean", "category"]).reset_index(drop=True)
    counts = df["category"].value_counts()
    rare = counts[counts < 10].index
    df = df[~df["category"].isin(rare)].reset_index(drop=True)
    print(f"Loaded {meta['rows_loaded']} rows -> {len(df)} after cleaning/dedup ({meta['schema']})")
    if df["category"].nunique() < 2 or len(df) < 50:
        raise SystemExit(
            f"Only {len(df)} usable rows / {df['category'].nunique()} categories left after cleaning, "
            "removing duplicates and dropping categories with fewer than 10 tickets. "
            "Check that --text-cols and --category-col point at the right columns.")
    has_prio = df["priority"].notna().any()
    if has_prio:
        df["priority"] = df["priority"].str.strip().str.title()

    # ---------------- 2. split ----------------
    train_idx, test_idx = train_test_split(df.index, test_size=a.test_size, stratify=df["category"], random_state=SEED)
    tr, te = df.loc[train_idx], df.loc[test_idx]
    classes = sorted(df["category"].unique())

    # ---------------- 3. compare models ----------------
    k = 3 if len(tr) > 20000 else 5
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    comparison = {}
    fitted = {}
    for name, pipe in make_models().items():
        t0 = time.time()
        scores = cross_val_score(pipe, tr["clean"], tr["category"], cv=cv, scoring="f1_macro")
        pipe.fit(tr["clean"], tr["category"])
        pred = pipe.predict(te["clean"])
        comparison[name] = {"cv_macro_f1_mean": float(scores.mean()), "cv_macro_f1_std": float(scores.std()),
                            "test": metrics_dict(te["category"], pred), "seconds": round(time.time() - t0, 1)}
        fitted[name] = pipe
        print(f"  {name:24s} CV macro-F1 {scores.mean():.3f} +/- {scores.std():.3f} | "
              f"test macro-F1 {comparison[name]['test']['macro_f1']:.3f} | {comparison[name]['seconds']}s")

    # model is chosen on cross-validation (NOT on the test set) to avoid test leakage
    best = max(comparison, key=lambda n: comparison[n]["cv_macro_f1_mean"])
    best_pipe = fitted[best]
    y_pred = best_pipe.predict(te["clean"])
    proba = best_pipe.predict_proba(te["clean"])
    conf = proba.max(axis=1)
    print(f"Selected model: {best}")

    plot_model_comparison(comparison, f"{a.out_dir}/model_comparison.png")
    plot_confusion(te["category"], y_pred, classes, f"Category confusion matrix - {best}",
                   f"{a.out_dir}/confusion_matrix_category.png")
    report = classification_report(te["category"], y_pred, output_dict=True, zero_division=0)
    terms = top_terms(tr["clean"], tr["category"])

    cm = confusion_matrix(te["category"], y_pred, labels=classes)
    off = cm.copy().astype(float)
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    most_confused = (classes[i], classes[j], int(off[i, j]))

    # confidence analysis -> supports the "human review" threshold
    thr = a.review_threshold
    hi = conf >= thr
    conf_stats = {
        "threshold": thr,
        "share_auto_routed": float(hi.mean()),
        "accuracy_auto_routed": float((y_pred[hi] == te["category"].values[hi]).mean()) if hi.any() else None,
        "accuracy_flagged_for_review": float((y_pred[~hi] == te["category"].values[~hi]).mean()) if (~hi).any() else None,
    }

    # ---------------- 4. priority ----------------
    prio_section, prio_pipe = {}, None
    rules_test = [score_ticket(t, c) for t, c in zip(te["raw_text"], y_pred)]
    if has_prio:
        # rows with a blank priority are skipped for the priority model only
        tr_p = tr[tr["priority"].notna()]
        mask_te = te["priority"].notna().values
        te_p = te[mask_te]
        rules_p = [r for r, m in zip(rules_test, mask_te) if m]
        prio_classes = sorted(df["priority"].dropna().unique(), key=lambda p: ORDER.index(p) if p in ORDER else 99)
        prio_pipe = Pipeline([("tfidf", tfidf()),
                              ("clf", LogisticRegression(max_iter=1000, C=5, class_weight="balanced"))])
        prio_pipe.fit(tr_p["clean"], tr_p["priority"])
        ml_prio = prio_pipe.predict(te_p["clean"])
        prio_section["n_priority_labelled_test"] = int(len(te_p))
        prio_section["ml_only"] = metrics_dict(te_p["priority"], ml_prio)
        if labels_compatible(prio_classes):
            rule_pred = [r.label for r in rules_p]
            hyb_pred = [hybrid_priority(m, r) for m, r in zip(ml_prio, rules_p)]
            prio_section["rules_only"] = metrics_dict(te_p["priority"], rule_pred)
            prio_section["hybrid"] = metrics_dict(te_p["priority"], hyb_pred)
            order_gap = np.abs(np.array([ORDER.index(p) for p in te_p["priority"]]) -
                               np.array([ORDER.index(p) for p in hyb_pred]))
            prio_section["hybrid_within_one_level"] = float((order_gap <= 1).mean())
            crit = te_p["priority"].values == "Critical"
            if crit.any():
                prio_section["critical_recall_ml_only"] = float((np.array(ml_prio)[crit] == "Critical").mean())
                prio_section["critical_recall_hybrid"] = float((np.array(hyb_pred)[crit] == "Critical").mean())
            plot_confusion(te_p["priority"], hyb_pred, prio_classes, "Priority confusion matrix - hybrid",
                           f"{a.out_dir}/confusion_matrix_priority.png")
        prio_section["classes"] = prio_classes
    else:
        prio_section["rule_distribution"] = dict(Counter(r.label for r in rules_test))

    # ---------------- 5. save ----------------
    bundle = {"category_model": best_pipe, "priority_model": prio_pipe, "meta": {
        **meta, "best_model": best, "classes": classes, "review_threshold": thr}}
    joblib.dump(bundle, f"{a.model_dir}/ticket_triage.joblib")

    all_metrics = {"dataset": meta, "n_rows_after_cleaning": len(df), "n_train": len(tr), "n_test": len(te),
                   "selected_model": best, "model_comparison": comparison,
                   "category_test_metrics": metrics_dict(te["category"], y_pred),
                   "per_class": {c: report[c] for c in classes}, "most_confused_pair": most_confused,
                   "confidence_analysis": conf_stats, "priority": prio_section}
    with open(f"{a.out_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # ---------------- 6. auto-generated report ----------------
    L = [f"# Evaluation report\n",
         f"*Dataset:* {meta['schema']} (`{os.path.basename(a.data)}`) - {len(df)} tickets after cleaning "
         f"({len(tr)} train / {len(te)} test, stratified split, seed {SEED}).\n",
         "## 1. Category model comparison\n",
         "Models are ranked by cross-validated macro-F1 on the training set; the test set is touched only once, "
         "after the winner is chosen.\n",
         md_table(["Model", "CV macro-F1", "Test accuracy", "Test macro-F1", "Time (s)"],
                  [[n + (" **(selected)**" if n == best else ""),
                    f"{c['cv_macro_f1_mean']:.3f} +/- {c['cv_macro_f1_std']:.3f}",
                    f"{c['test']['accuracy']:.3f}", f"{c['test']['macro_f1']:.3f}", c["seconds"]]
                   for n, c in comparison.items()]),
         "\n![model comparison](model_comparison.png)\n",
         f"## 2. Selected model: {best}\n",
         md_table(["Category", "Precision", "Recall", "F1", "Support"],
                  [[c, f"{report[c]['precision']:.3f}", f"{report[c]['recall']:.3f}",
                    f"{report[c]['f1-score']:.3f}", int(report[c]['support'])] for c in classes]),
         f"\nOverall test accuracy **{all_metrics['category_test_metrics']['accuracy']:.3f}**, "
         f"macro-F1 **{all_metrics['category_test_metrics']['macro_f1']:.3f}**.\n",
         "![confusion matrix](confusion_matrix_category.png)\n",
         f"Most frequent confusion: **{most_confused[0]} -> predicted as {most_confused[1]}** "
         f"({most_confused[2]} tickets).\n",
         "### Most indicative words per category\n",
         md_table(["Category", "Top words"], [[c, ", ".join(w)] for c, w in terms.items()]),
         "\n### Confidence and human review\n",
         f"Tickets with confidence >= {thr:.0%} are auto-routed: **{conf_stats['share_auto_routed']:.0%}** of the test set, "
         f"with accuracy **{conf_stats['accuracy_auto_routed']:.3f}**." +
         (f" Tickets below the threshold are flagged for a human; accuracy there is "
          f"{conf_stats['accuracy_flagged_for_review']:.3f}.\n" if conf_stats["accuracy_flagged_for_review"] is not None else "\n"),
         "## 3. Priority\n"]
    if has_prio:
        rows = [[k.replace('_', ' '), f"{v['accuracy']:.3f}", f"{v['macro_f1']:.3f}"]
                for k, v in prio_section.items() if isinstance(v, dict) and 'accuracy' in v]
        L += [md_table(["System", "Accuracy", "Macro-F1"], rows) + "\n"]
        if "hybrid_within_one_level" in prio_section:
            L.append(f"Hybrid priority is within one level of the true label for "
                     f"**{prio_section['hybrid_within_one_level']:.0%}** of tickets.\n")
        if "critical_recall_hybrid" in prio_section:
            L.append(f"Recall on *Critical* tickets: ML only **{prio_section['critical_recall_ml_only']:.2f}** vs "
                     f"hybrid **{prio_section['critical_recall_hybrid']:.2f}**.\n")
        if os.path.exists(f"{a.out_dir}/confusion_matrix_priority.png"):
            L.append("![priority confusion](confusion_matrix_priority.png)\n")
    else:
        L.append("No priority labels in this dataset, so the rule engine alone decides priority. "
                 f"Distribution on the test set: {prio_section['rule_distribution']}\n")
    with open(f"{a.out_dir}/REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nSaved model to {a.model_dir}/ and report to {a.out_dir}/REPORT.md")


if __name__ == "__main__":
    main()
