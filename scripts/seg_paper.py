import pandas as pd, json, sys
sys.path.insert(0,".")
from src.parser.content_parser import parse_paper_content
df = pd.read_parquet("data/train-00000-of-00001.parquet")
row = df[df["doi/arxiv_id"].astype(str) == "2406.11623v4"].iloc[0]
pid = str(row["doi/arxiv_id"])
paper = parse_paper_content(paper_id=pid, title=row["title"], paper_category=row["paper_category"], paper_content=row["paper_content"], decode_images=False)

snips = []
for s in paper.sections:
    snips.append(dict(snippet_id=s.id, snippet_type="SECTION", paper_id=pid, location=s.section_title, content=s.content, metadata=dict(section_level=s.section_level)))

for t in paper.theorems:
    c = t.statement
    if t.proof:
        c = c + "

Proof: " + t.proof
    snips.append(dict(snippet_id=t.id, snippet_type=t.theorem_type.upper(), paper_id=pid, location=t.label or t.id, content=c, metadata=dict(theorem_type=t.theorem_type, has_proof=t.proof is not None)))

eqs = paper.equations
bs = 10
for i in range(0, len(eqs), bs):
    batch = eqs[i:i+bs]
    bid = pid + "_eq_batch_" + str(i//bs)
    parts = []
    for j, eq in enumerate(batch):
        lbl = eq.equation_label or ("unnamed_" + str(j))
        parts.append("Equation " + lbl + ":
" + eq.latex)
    end = min(i+bs, len(eqs))
    snips.append(dict(snippet_id=bid, snippet_type="EQUATION", paper_id=pid, location="Eq batch " + str(i//bs) + " (" + str(i+1) + "-" + str(end) + ")", content="

".join(parts), metadata=dict(equation_indices=list(range(i, end)), batch_size=len(batch))))

json.dump(snips, open("tmp_segmented.json", "w"), indent=2)
print("Segmented:", len(snips), "snippets")
