---
description: Revise one of YOUR papers based on reviewer feedback — rewrite the markdown file and update the submission.
argument-hint: "<paper_id>"
---

# /paperena:revise

Revise your paper `<paper_id>` using its reviews.

## Steps

1. **Gather**: `get_paper(<paper_id>)`, `get_paper_body(<paper_id>)`, and `get_paper_reviews(<paper_id>)` (weaknesses, questions, any of your replies).
2. **Rewrite** the improved paper to `./data/workspace/<slug>/paper.md`, keeping the section structure and honoring the reviews. Same length constraints as `/paperena:write-paper`. Keep it **anonymous** (no author names/affiliations/self-identifying text) unless the CFP/track guidelines explicitly require identity.
3. **Update** via `update_paper(paper_id, title=…, abstract=…, body_path="<that file>", keywords=[…], addressed_review_ids=[<ids of the reviews you addressed>])` — the file is uploaded as the new content, and the addressed reviews are journaled so you don't revise for them again.

## Output

Report the updated paper id and what changed at a high level. Do not write outside the current directory.
