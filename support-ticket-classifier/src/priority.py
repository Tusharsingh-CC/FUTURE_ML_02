"""Explainable priority scoring for support tickets.

Priority is decided by a transparent point system. Every point comes with a
human-readable reason, so a support manager can see *why* a ticket was
escalated. It works on the RAW text (not the cleaned text) because capital
letters and exclamation marks are useful frustration signals.

Final priority used by the system (see `hybrid_priority`):
  * If the dataset has priority labels -> an ML model predicts priority, and the
    rule engine can only ESCALATE (never lower) when it finds a hard-critical
    signal such as an outage, data loss or fraud.
  * If the dataset has NO priority labels -> the rule engine alone decides.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

ORDER = ["Low", "Medium", "High", "Critical"]
RANK = {p: i for i, p in enumerate(ORDER)}

# ---- signal dictionaries -------------------------------------------------
# (regex, label)  -> instant escalation to Critical
HARD_ESCALATION = [
    (r"\b(outage|(system|server|site|website|production|service|app|platform) (is |are )?(completely |totally )?down)\b", "service outage"),
    (r"\b(data (loss|breach|leak)|lost (all )?(of )?(the |my |our )?data|data (is )?(gone|deleted))\b", "data loss / breach"),
    (r"\b(hacked|breach(ed)?|compromised|fraud(ulent)?|unauthori[sz]ed (access|charge|transaction|login))\b", "security / fraud"),
]

STRONG_URGENCY = [
    r"\burgent(ly)?\b", r"\basap\b", r"\bimmediate(ly)?\b", r"\bemergency\b",
    r"\bcritical\b", r"\bright now\b", r"\bblock(ed|ing)\b", r"\bas soon as possible\b",
    r"\b(unable to|cannot|can ?not) (work|operate|do my job)\b", r"\bdeadline\b",
    r"\blosing (money|customers|revenue|business)\b", r"\btoday\b",
]
FAILURE = [
    r"\bnot working\b", r"\b(does not|doesn't|do not|don't) work\b", r"\bstopped working\b",
    r"\bcrash(es|ed|ing)?\b", r"\bfail(ed|s|ing|ure)?\b", r"\berror\b", r"\bbroken\b",
    r"\bfreez(e|es|ing)\b", r"\b(cannot|can't|unable to)\b",
]
FRUSTRATION = [
    r"\bunacceptable\b", r"\bfrustrat", r"\bfurious\b", r"\bangry\b", r"\bworst\b",
    r"\bterrible\b", r"\bridiculous\b", r"\bdisappointed\b", r"\b(second|third|fourth) time\b",
    r"\bstill (not|no)\b", r"\bfed up\b",
]
MONEY_IMPACT = [
    r"\bcharged (me )?(twice|double)\b", r"\bdouble[- ]charged\b", r"\bovercharg",
    r"\bwrong(ly)? charged\b", r"\bmoney (was )?(taken|deducted)\b",
]
LOW_URGENCY = [
    r"\bno rush\b", r"\bwhenever\b", r"\bjust (wondering|curious|a (quick )?question)\b",
    r"\bquick question\b", r"\bwhen you (get|have) (a chance|time)\b",
    r"\bat your convenience\b", r"\bnonurgent\b", r"\bno hurry\b",
]

# category-name hints (matched as substrings, so it works with any dataset)
CATEGORY_BOOST = [
    (("technical", "outage", "access", "security", "hardware", "network", "server", "login"), 1),
    (("billing", "refund", "payment", "invoice"), 1),
]
CATEGORY_LOWER = ("product inquiry", "general", "query", "information", "misc")

# "we had NO data loss" / "was not hacked" must not trigger a hard escalation
_NEGATED = re.compile(r"(\b(no|not|never|without|zero)\b|n't\b)[\w\s]{0,15}$")


@dataclass
class PriorityResult:
    score: int
    label: str
    hard_escalation: bool
    reasons: List[str] = field(default_factory=list)


def _hits(patterns, text) -> List[str]:
    found = []
    for p in patterns:
        m = re.search(p, text)
        if m:
            found.append(m.group(0).strip())
    return found


def score_ticket(raw_text: str, category: Optional[str] = None) -> PriorityResult:
    raw = raw_text if isinstance(raw_text, str) else ""
    t = raw.lower()
    t = re.sub(r"\bnot (very )?urgent\b", " nonurgent ", t)
    score, reasons = 0, []

    # 1) hard escalation: instantly Critical
    hard = []
    for pat, name in HARD_ESCALATION:
        for m in re.finditer(pat, t):
            if _NEGATED.search(t[max(0, m.start() - 25):m.start()]):
                continue  # negated mention, e.g. "no data loss"
            hard.append(name)
            break
    if hard:
        score += 6
        reasons.append(f"+6 hard-critical signal ({', '.join(hard)})")

    # 2) soft signals, each capped so one noisy group cannot dominate
    for patterns, weight, cap, name in [
        (STRONG_URGENCY, 2, 4, "urgency words"),
        (FAILURE, 1, 2, "failure words"),
        (FRUSTRATION, 1, 2, "frustration words"),
        (MONEY_IMPACT, 2, 2, "money impact"),
    ]:
        found = _hits(patterns, t)
        if found:
            pts = min(weight * len(found), cap)
            score += pts
            reasons.append(f"+{pts} {name}: {', '.join(found[:3])}")

    # 3) tone signals from raw text
    if raw.count("!") >= 2:
        score += 1
        reasons.append("+1 multiple exclamation marks")
    words = re.findall(r"\b[A-Za-z]{3,}\b", raw)
    if len(words) >= 4 and sum(w.isupper() for w in words) / len(words) > 0.15:
        score += 1
        reasons.append("+1 shouting (many CAPITAL words)")

    # 4) low-urgency politeness cues
    low = _hits(LOW_URGENCY, t)
    if low:
        score -= 2
        reasons.append(f"-2 low-urgency wording: {', '.join(low[:2])}")

    # 5) category weight
    if category:
        c = str(category).lower()
        if any(k in c for keys, _ in CATEGORY_BOOST for k in keys):
            score += 1
            reasons.append(f"+1 category '{category}' usually impacts the customer directly")
        elif any(k in c for k in CATEGORY_LOWER):
            score -= 1
            reasons.append(f"-1 category '{category}' is usually informational")

    score = max(score, 0)
    if score >= 6:
        label = "Critical"
    elif score >= 4:
        label = "High"
    elif score >= 2:
        label = "Medium"
    else:
        label = "Low"
    return PriorityResult(score, label, bool(hard), reasons)


def labels_compatible(labels) -> bool:
    """True if a dataset's priority labels can be mapped onto Low/Medium/High/Critical."""
    return {str(l).strip().title() for l in labels} <= set(ORDER)


def hybrid_priority(ml_label: Optional[str], rule: PriorityResult) -> str:
    """Combine ML and rules. The rules may only escalate, never downgrade."""
    if ml_label is None:
        return rule.label
    ml_label = str(ml_label).strip().title()
    if ml_label not in RANK:
        return ml_label
    if rule.hard_escalation:
        return "Critical"
    return ml_label
