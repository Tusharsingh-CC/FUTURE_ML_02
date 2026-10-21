"""Generate a small SYNTHETIC ticket file with the same columns as the Kaggle
'Customer Support Ticket Dataset'. It exists only so the pipeline can be tested
end-to-end without downloading anything. Do NOT report its metrics as real results.

Usage: python -m src.make_sample_data --rows 6000 --out data/sample_tickets.csv
"""
import argparse
import random

import pandas as pd

PRODUCTS = ["Dell XPS 13", "GoPro Hero", "Sony 4K TV", "Canon EOS", "Amazon Echo", "AirPods",
            "Nest Thermostat", "Fitbit Charge", "LG OLED TV", "Philips Hue", "Microsoft Office", "Adobe Photoshop"]

CORE = {
    "Technical issue": [
        "my {p} keeps crashing every time I open it",
        "the {p} will not turn on after the latest update",
        "I get an error message when I connect my {p} to wifi",
        "{p} is freezing and the screen goes black randomly",
        "I cannot log in to the app linked to my {p}, it says authentication failed",
        "the sync feature on my {p} stopped working",
        "battery on my {p} drains within an hour",
    ],
    "Billing inquiry": [
        "I was charged twice for my {p} subscription this month",
        "there is an unknown charge on my invoice related to {p}",
        "my latest bill is higher than the amount I agreed to pay for {p}",
        "I need a copy of the invoice for my {p} purchase",
        "the discount code was not applied to my {p} order",
        "can you explain the tax on my {p} invoice",
    ],
    "Cancellation request": [
        "I would like to cancel my {p} subscription",
        "please close my account and stop billing me for {p}",
        "I want to terminate my {p} plan at the end of this month",
        "how do I cancel the auto renewal for {p}",
        "I no longer need {p} and want to cancel",
    ],
    "Product inquiry": [
        "does the {p} support bluetooth 5 and wireless charging",
        "what is the warranty period for the {p}",
        "is the {p} compatible with my existing setup",
        "what is the difference between the {p} and the newer model",
        "is the {p} available in other colors",
        "where can I find the user manual for {p}",
    ],
    "Refund request": [
        "I want a refund for my {p}, it arrived damaged",
        "the {p} is not what I ordered, please refund my money",
        "I returned the {p} two weeks ago but have not received my refund",
        "please process a refund for the defective {p}",
        "the {p} does not work as advertised and I want my money back",
    ],
}
SUBJECTS = {
    "Technical issue": ["Product not working", "App problem", "Error on device", "Technical support needed"],
    "Billing inquiry": ["Billing question", "Charge on my account", "Invoice issue", "Payment problem"],
    "Cancellation request": ["Cancel my plan", "Cancellation", "Close my account"],
    "Product inquiry": ["Question about product", "Product information", "Compatibility question"],
    "Refund request": ["Refund needed", "Return and refund", "Money back"],
}
FILLER = ["I bought it last month.", "I have contacted support before.", "Please help.",
          "My order number is {n}.", "Thank you for your time.", "I am a long-time customer."]

URGENCY = {
    "Critical": ["This is extremely URGENT, our whole team is blocked and we are losing money!!",
                 "We have had a complete outage and lost data, need immediate help!",
                 "I think my account was hacked, this is a security issue."],
    "High": ["Please fix this as soon as possible.", "I need this resolved today, I have a deadline.",
             "This is the third time I am writing about this, it is unacceptable."],
    "Medium": ["Please look into it this week.", "It is affecting my work a bit.", ""],
    "Low": ["No rush, just wondering.", "Whenever you get a chance is fine.", "Just a quick question.", ""],
}
PRIORITY_BY_TYPE = {  # probabilities for Low, Medium, High, Critical
    "Technical issue": [0.15, 0.30, 0.35, 0.20],
    "Billing inquiry": [0.25, 0.40, 0.25, 0.10],
    "Cancellation request": [0.30, 0.40, 0.25, 0.05],
    "Product inquiry": [0.60, 0.30, 0.10, 0.00],
    "Refund request": [0.20, 0.40, 0.30, 0.10],
}
LEVELS = ["Low", "Medium", "High", "Critical"]


def make_ticket(rng, i):
    ttype = rng.choice(list(CORE))
    p = rng.choice(PRODUCTS)
    prio = rng.choices(LEVELS, PRIORITY_BY_TYPE[ttype])[0]

    parts = [rng.choice(CORE[ttype]).format(p=p).capitalize() + "."]
    if rng.random() < 0.15:  # mixed-intent ticket -> realistic confusion between classes
        other = rng.choice([t for t in CORE if t != ttype])
        parts.append("Also, " + rng.choice(CORE[other]).format(p=p) + ".")
    parts.append(rng.choice(FILLER).format(n=rng.randint(10000, 99999)))
    if rng.random() < 0.85:
        level = prio if rng.random() > 0.15 else rng.choice(LEVELS)  # 15% noisy urgency wording
        phrase = rng.choice(URGENCY[level])
        if phrase:
            parts.append(phrase)

    # label noise: a few tickets are mis-labelled, as in real life
    label = ttype if rng.random() > 0.05 else rng.choice(list(CORE))
    return {
        "Ticket ID": i + 1,
        "Ticket Subject": rng.choice(SUBJECTS[ttype]),
        "Ticket Description": " ".join(parts),
        "Ticket Type": label,
        "Ticket Priority": prio,
        "Ticket Channel": rng.choice(["Email", "Chat", "Phone", "Social media"]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--out", default="data/sample_tickets.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pd.DataFrame([make_ticket(rng, i) for i in range(a.rows)]).to_csv(a.out, index=False)
    print(f"Wrote {a.rows} synthetic tickets to {a.out}")
