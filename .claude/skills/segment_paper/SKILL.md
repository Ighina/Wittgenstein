---
name: segment-markdown-paper
description: "Use this skill when you have a paper already expressed as a markdown string and need to segment it into typed verification snippets (SECTION, EQUATION, FIGURE, TABLE, THEOREM, LEMMA, PROOF, ALGORITHM, CODE) without calling any MCP parsing tool first. Input is raw markdown text; output is a structured list of snippets ready for per-snippet verification."
compatibility: "Claude Code, claude.ai, Claude Desktop"
---

# Segment a Markdown Paper into Verification Snippets

## Input contract

| Field | Type | Description |
|---|---|---|
| `paper_id` | string | DOI or arXiv ID, used to prefix snippet IDs |
| `title` | string | Paper title |
| `paper_category` | string | Scientific field, e.g. `"cs.LG"`, `"math.ST"` |
| `markdown` | string | Full paper body as markdown |

---

## Output schema

Produce a JSON object with this shape:

```json
{
  "paper_id": "<paper_id>",
  "title": "<title>",
  "paper_category": "<paper_category>",
  "total_snippets": 12,
  "snippets": [
    {
      "snippet_id": "<paper_id>::SECTION::0",
      "snippet_type": "SECTION",
      "location": "Introduction",
      "content": "...",
      "content_length": 843,
      "metadata": {}
    }
  ]
}
```

Snippet IDs follow the pattern `<paper_id>::<TYPE>::<index>` where index is
the zero-based counter **within that type**.

---

## Snippet types and how to recognise them in markdown

Work through the markdown **top-to-bottom in a single pass**, collecting
snippets in document order.

### SECTION

A run of prose text under a heading. Emit one SECTION snippet per heading.

- **Trigger**: `#`, `##`, `###`, `####` headings.
- **`location`**: the heading text, e.g. `"3.2 Convergence Analysis"`.
- **`content`**: all prose lines that follow the heading, up to (but not
  including) the next heading of equal or higher level, and excluding any
  embedded blocks that are captured as their own snippet types below.
- **`metadata`**: `{ "heading_level": 2 }`.

### EQUATION

A display-mode LaTeX block.

- **Trigger**: fenced blocks ` ```math ` … ` ``` `, or `$$` … `$$` spans, or
  `\begin{equation}` / `\begin{align}` / `\begin{gather}` environments.
- **`location`**: the heading of the containing section.
- **`content`**: the raw LaTeX source including delimiters.
- **`metadata`**: `{ "label": "<label if present, else null>", "display_mode": true }`.
- Inline math (`$…$`) is **not** a standalone snippet; leave it inside its
  SECTION.

### FIGURE

An image reference or figure environment.

- **Trigger**: `![caption](url)` markdown images, or `\begin{figure}` …
  `\end{figure}` blocks.
- **`location`**: containing section heading.
- **`content`**: the full figure block including caption text.
- **`metadata`**: `{ "caption": "<caption text or null>" }`.

### TABLE

A markdown table or LaTeX tabular environment.

- **Trigger**: pipe-table syntax (`| col | col |` with a separator row), or
  `\begin{table}` / `\begin{tabular}` blocks.
- **`location`**: containing section heading.
- **`content`**: the full table block.
- **`metadata`**: `{ "caption": "<caption if present, else null>", "row_count": <int> }`.
  Count rows by counting `|`-delimited data rows (exclude the separator row).

### THEOREM / LEMMA / PROPOSITION / COROLLARY / DEFINITION

Formal mathematical statements.

- **Trigger**: bold or heading lines matching
  `**Theorem N**`, `**Lemma N**`, `**Proposition N**`, `**Corollary N**`,
  `**Definition N**`; or `\begin{theorem}` / `\begin{lemma}` / etc.
- **`snippet_type`**: use the matched keyword in ALL CAPS (`THEOREM`, `LEMMA`,
  `PROPOSITION`, `COROLLARY`, `DEFINITION`).
- **`location`**: containing section heading.
- **`content`**: the statement text through the end of the environment or
  blank-line boundary.
- **`metadata`**: `{ "label": "<e.g. Theorem 3.1 or null>", "has_proof": <bool> }`.
  Set `has_proof: true` when a PROOF snippet immediately follows.

### PROOF

A proof block.

- **Trigger**: `**Proof.**`, `*Proof.*`, `\begin{proof}`, or a paragraph
  starting with `Proof.` or `Proof of`.
- **`location`**: containing section heading.
- **`content`**: all text through `\end{proof}`, `□`, `∎`, or `QED` (inclusive).
- **`metadata`**: `{ "of": "<label of the theorem this proves, or null>" }`.

### ALGORITHM

A pseudocode or algorithm block.

- **Trigger**: fenced code blocks tagged ` ```algorithm `, or
  `\begin{algorithm}` environments, or headings/bold lines containing
  `Algorithm N`.
- **`location`**: containing section heading.
- **`content`**: the full block.
- **`metadata`**: `{ "label": "<Algorithm N or null>" }`.

### CODE

A source-code listing.

- **Trigger**: fenced code blocks with a language tag other than `math` or
  `algorithm` (e.g. ` ```python `, ` ```julia `), or untagged fenced blocks
  that contain code-like content.
- **`location`**: containing section heading.
- **`content`**: the full fenced block.
- **`metadata`**: `{ "language": "<tag or null>" }`.

---

## Segmentation procedure

```
state:
  current_section_heading  ← null
  current_section_lines    ← []
  counters                 ← { SECTION:0, EQUATION:0, … }  # per type
  snippets                 ← []

for each line (and multi-line block) in markdown:

  if line is a heading:
    flush current_section_lines as a SECTION snippet (if non-empty)
    current_section_heading ← heading text
    current_section_lines   ← []

  elif line opens a display equation / table / figure / theorem /
       proof / algorithm / code block:
    consume lines until block closes
    emit the appropriate typed snippet
    (do NOT add these lines to current_section_lines)

  else:
    append line to current_section_lines

# after loop: flush any remaining section lines
flush current_section_lines as SECTION snippet (if non-empty)
```

Truncate `content` to **3000 characters** in each snippet; record the
original length in `content_length`.

---

## Handling ambiguous or malformed markdown

- If a block trigger is opened but never closed (e.g. unclosed `$$`), close
  it at the next blank line and add `"malformed": true` to `metadata`.
- If there is no heading at all before the first content, use
  `location: "Preamble"`.
- Nested LaTeX environments: match `\begin{X}` / `\end{X}` pairs by name,
  not just nesting depth.

---

## Example (abbreviated)

Given this markdown fragment:

```markdown
## 3 Method

We propose a new estimator based on the following key result.

**Theorem 3.1** (Unbiasedness). Let $X_1, \ldots, X_n$ be i.i.d. Then

$$
\hat{\mu} = \frac{1}{n}\sum_{i=1}^n X_i
$$

is an unbiased estimator of $\mu$.

**Proof.** $\mathbb{E}[\hat{\mu}] = \mu$ by linearity of expectation. □
```

Produce:

```json
[
  {
    "snippet_id": "2401.00001::SECTION::0",
    "snippet_type": "SECTION",
    "location": "3 Method",
    "content": "We propose a new estimator based on the following key result.",
    "content_length": 61,
    "metadata": { "heading_level": 2 }
  },
  {
    "snippet_id": "2401.00001::THEOREM::0",
    "snippet_type": "THEOREM",
    "location": "3 Method",
    "content": "**Theorem 3.1** (Unbiasedness). Let $X_1, \\ldots, X_n$ be i.i.d. Then is an unbiased estimator of $\\mu$.",
    "content_length": 104,
    "metadata": { "label": "Theorem 3.1", "has_proof": true }
  },
  {
    "snippet_id": "2401.00001::EQUATION::0",
    "snippet_type": "EQUATION",
    "location": "3 Method",
    "content": "$$\n\\hat{\\mu} = \\frac{1}{n}\\sum_{i=1}^n X_i\n$$",
    "content_length": 46,
    "metadata": { "label": null, "display_mode": true }
  },
  {
    "snippet_id": "2401.00001::PROOF::0",
    "snippet_type": "PROOF",
    "location": "3 Method",
    "content": "**Proof.** $\\mathbb{E}[\\hat{\\mu}] = \\mu$ by linearity of expectation. □",
    "content_length": 71,
    "metadata": { "of": "Theorem 3.1" }
  }
]
```

---

## Output instructions

- Emit the full JSON object described in **Output schema** above.
- Do not summarise or truncate the `snippets` array; include every snippet.
- If the markdown is empty or contains only whitespace, return
  `{ "total_snippets": 0, "snippets": [] }` with the paper metadata fields
  still populated.
- Do not add commentary outside the JSON block.