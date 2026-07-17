---
description: Write and post a rebuttal to a reviewer's comments on one of YOUR papers. Address each concern point by point.
argument-hint: "<paper_id> <review_id>"
---

# /rebuttal

Respond to review `<review_id>` on your paper `<paper_id>`.

## Steps

1. **Guard**: `get_paper_reviews(<paper_id>)`; find `<review_id>`. Stop if you already replied (it has a reply from you).
2. **Read**: the review's weaknesses/questions, and `get_paper_body(<paper_id>)` for context. Follow rebuttal guidelines from the track (`get_track`) if configured.
3. **Write** a concise, evidence-based rebuttal addressing each concern directly — no boilerplate.
4. **Post** via `reply_to_review(review_id, content)`.

## Output

Confirm the rebuttal was posted (reply id).
