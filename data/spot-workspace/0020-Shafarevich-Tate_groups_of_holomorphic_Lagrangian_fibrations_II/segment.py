#!/usr/bin/env python3
"""Segment the Shafarevich-Tate paper into verification snippets."""
import json, re, sys

with open(sys.argv[1], 'r') as f:
    text = f.read()

lines = text.split('\n')
paper_id = "0020-Shafarevich-Tate_II"
title = "Shafarevich-Tate Groups of Holomorphic Lagrangian Fibrations II"
paper_category = "math.AG"

snippets = []
counters = {"SECTION":0, "EQUATION":0, "FIGURE":0, "TABLE":0,
            "THEOREM":0, "LEMMA":0, "PROPOSITION":0, "COROLLARY":0,
            "DEFINITION":0, "PROOF":0, "ALGORITHM":0, "CODE":0,
            "REMARK":0}

current_section_heading = "Preamble"
current_section_lines = []
i = 0

def flush_section():
    global current_section_lines
    content = '\n'.join(current_section_lines).strip()
    if content:
        counters["SECTION"] += 1
        snippets.append({
            "snippet_id": f"{paper_id}::SECTION::{counters['SECTION']-1}",
            "snippet_type": "SECTION",
            "location": current_section_heading,
            "content": content[:3000],
            "content_length": len(content),
            "metadata": {"heading_level": current_section_heading.count('#') if current_section_heading.startswith('#') else 1}
        })
    current_section_lines = []

def make_snippet(stype, content, location, extra_meta=None):
    counters[stype] += 1
    meta = extra_meta or {}
    if stype in ("THEOREM","LEMMA","PROPOSITION","COROLLARY","DEFINITION"):
        # Extract label
        m = re.match(r'\*\*([^*]+)\*\*', content)
        meta["label"] = m.group(1).strip() if m else None
        meta["has_proof"] = False
    elif stype == "EQUATION":
        meta["display_mode"] = True
        meta["label"] = None
    elif stype == "PROOF":
        meta["of"] = content.split('.')[0] if '.' in content[:50] else None
    elif stype == "FIGURE":
        m = re.search(r'\[(.*?)\]', content)
        meta["caption"] = m.group(1) if m else None

    snippets.append({
        "snippet_id": f"{paper_id}::{stype}::{counters[stype]-1}",
        "snippet_type": stype,
        "location": location,
        "content": content[:3000],
        "content_length": len(content),
        "metadata": meta
    })

# Regex patterns
heading_pat = re.compile(r'^(#{1,4})\s+(.+)')
definition_pat = re.compile(r'^\*\*Definition\s+[\d.]+\*\*')
theorem_pat = re.compile(r'^\*\*Theorem\s+[A-E]\b')
theorem_letter_pat = re.compile(r'^\*\*Theorem\s+([A-E])\b')
theorem_num_pat = re.compile(r'^\*\*Theorem\s+([\d.]+)\*\*')
lemma_pat = re.compile(r'^\*\*Lemma\s+([\d.]+)\*\*')
proposition_pat = re.compile(r'^\*\*Proposition\s+([\d.]+)\*\*')
corollary_pat = re.compile(r'^\*\*Corollary\s+([\d.]+)\*\*')
proof_pat = re.compile(r'^(Proof\.|\*\*Proof\.\*\*|\*Proof\.\*)')
display_math_open = re.compile(r'^\s*(\$\$|\\\[|\\begin\{(equation|align|gather|array)\})')
display_math_close = re.compile(r'(\$\$|\\\]|\\end\{(equation|align|gather|array)\})')
figure_pat = re.compile(r'^!\[.*?\]\(.*?\)')

in_display_math = False
math_lines = []
in_proof = False
proof_lines = []

i = 0
while i < len(lines):
    line = lines[i]

    # Heading
    hm = heading_pat.match(line)
    if hm:
        flush_section()
        current_section_heading = hm.group(2).strip()
        i += 1
        continue

    # Display math start
    dm_open = display_math_open.match(line)
    if dm_open and not in_display_math:
        in_display_math = True
        math_lines = [line]
        i += 1
        continue

    # Inside display math
    if in_display_math:
        math_lines.append(line)
        if display_math_close.search(line) or (dm_open and line.strip().endswith('$$') and len(math_lines) > 1):
            in_display_math = False
            make_snippet("EQUATION", '\n'.join(math_lines), current_section_heading)
        i += 1
        continue

    # Figure
    if figure_pat.match(line):
        fig_content = line
        i += 1
        while i < len(lines) and lines[i].strip() and not heading_pat.match(lines[i]):
            if lines[i].strip().startswith('http') or lines[i].strip().startswith('data:'):
                fig_content += '\n' + lines[i]
            i += 1
        make_snippet("FIGURE", fig_content, current_section_heading)
        continue

    # Definition
    if definition_pat.match(line):
        def_content = line
        i += 1
        while i < len(lines) and lines[i].strip() and not heading_pat.match(lines[i]) and not re.match(r'^\*\*(Definition|Theorem|Lemma|Proposition|Corollary)', lines[i]):
            if not re.match(r'^\^', lines[i]):  # skip footnotes
                def_content += '\n' + lines[i]
            i += 1
        make_snippet("DEFINITION", def_content, current_section_heading)
        continue

    # Theorem (letter-based like Theorem A, B, C, D)
    if theorem_letter_pat.match(line):
        thm_content = line
        i += 1
        while i < len(lines) and lines[i].strip() and not heading_pat.match(lines[i]) and not re.match(r'^\*\*(Definition|Theorem|Lemma|Proposition|Corollary)', lines[i]) and not proof_pat.match(lines[i]):
            thm_content += '\n' + lines[i]
            i += 1
        make_snippet("THEOREM", thm_content, current_section_heading)
        continue

    # Theorem (number-based)
    if theorem_num_pat.match(line):
        thm_content = line
        i += 1
        while i < len(lines) and lines[i].strip() and not heading_pat.match(lines[i]) and not re.match(r'^\*\*(Definition|Theorem|Lemma|Proposition|Corollary)', lines[i]) and not proof_pat.match(lines[i]):
            thm_content += '\n' + lines[i]
            i += 1
        make_snippet("THEOREM", thm_content, current_section_heading)
        continue

    # Lemma
    if lemma_pat.match(line):
        lem_content = line
        i += 1
        while i < len(lines) and lines[i].strip() and not heading_pat.match(lines[i]) and not re.match(r'^\*\*(Definition|Theorem|Lemma|Proposition|Corollary)', lines[i]) and not proof_pat.match(lines[i]):
            lem_content += '\n' + lines[i]
            i += 1
        make_snippet("LEMMA", lem_content, current_section_heading)
        continue

    # Proposition
    if proposition_pat.match(line):
        prop_content = line
        i += 1
        while i < len(lines) and lines[i].strip() and not heading_pat.match(lines[i]) and not re.match(r'^\*\*(Definition|Theorem|Lemma|Proposition|Corollary)', lines[i]) and not proof_pat.match(lines[i]):
            prop_content += '\n' + lines[i]
            i += 1
        make_snippet("PROPOSITION", prop_content, current_section_heading)
        continue

    # Corollary
    if corollary_pat.match(line):
        cor_content = line
        i += 1
        while i < len(lines) and lines[i].strip() and not heading_pat.match(lines[i]) and not re.match(r'^\*\*(Definition|Theorem|Lemma|Proposition|Corollary)', lines[i]) and not proof_pat.match(lines[i]):
            cor_content += '\n' + lines[i]
            i += 1
        make_snippet("COROLLARY", cor_content, current_section_heading)
        continue

    # Proof
    if proof_pat.match(line):
        proof_content = line
        i += 1
        while i < len(lines):
            if lines[i].strip() in ('□', '∎') or re.match(r'^\*\*(Definition|Theorem|Lemma|Proposition|Corollary)', lines[i]) or heading_pat.match(lines[i]):
                if lines[i].strip() in ('□', '∎'):
                    proof_content += '\n' + lines[i]
                    i += 1
                break
            proof_content += '\n' + lines[i]
            i += 1
        make_snippet("PROOF", proof_content, current_section_heading)
        continue

    # Regular text - accumulate
    current_section_lines.append(line)
    i += 1

# Flush remaining section
flush_section()

# Build output
output = {
    "paper_id": paper_id,
    "title": title,
    "paper_category": paper_category,
    "total_snippets": len(snippets),
    "snippets": snippets
}

with open(sys.argv[2], 'w') as f:
    json.dump(output, f, indent=2)

print(f"Segmented into {len(snippets)} snippets:")
for stype in sorted(counters.keys()):
    if counters[stype] > 0:
        print(f"  {stype}: {counters[stype]}")
