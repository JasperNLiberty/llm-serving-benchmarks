# Experiment 03 — The Business Story

*A hypothetical to make the numbers matter. Fictional company; the data is real (see [analysis.md](analysis.md)).*

## The company
**Tassel** is an e-commerce platform. Their AI does a grab-bag of jobs: a lot of **simple, high-volume** ones (classify a support email, extract an order number, route a ticket) and a few **hard** ones (write a personalized recommendation blurb). Engineering standardized on one model — the 7B — "so we only run one thing."

## The decision on the table
**One model for everything, or right-size per task?** Running a single 7B is operationally simple. What does that simplicity cost when most of the volume is easy work?

## What the experiment tells them
The 7B is **~2.2× the cost per token** of the 1B — and the high-volume jobs (classification, extraction, routing) are exactly the ones a 1B handles fine. Standardizing on the 7B means paying a **~2× tax on the bulk of the traffic** for capability it doesn't use.

![cost by model](screenshots/03-cost_per_million_tokens_by_model.png)

The flip side, and the reason not to just use the 1B everywhere: the premium is *only* ~2×, not 7×, so when a task genuinely needs the bigger model, upgrading is far cheaper than its size suggests. The cost curve rewards **matching model to task**, not picking one end.

## The recommendation
> **Right-size per task, not per company.** Route the high-volume simple jobs to the 1B and reserve the 7B for the work that needs it. Since simple jobs dominate the volume, moving them to the 1B roughly **halves the cost of that traffic** at no quality loss — while the modest 2× premium means the hard jobs stay affordable on the big model. "One model to rule them all" is operational convenience bought with a standing ~2× overpay on your most common requests.

## The one-sentence executive summary
> A 7B model costs only ~2× a 1B (not 7×), so the economical play is to **route by task difficulty** — cheap model for the high-volume simple work, big model reserved for the hard minority — rather than standardizing on one size and overpaying on the bulk of traffic.
