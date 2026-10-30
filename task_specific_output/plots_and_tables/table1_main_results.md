# Table 1 — Mathematical-error detection on PaperAudit and SPOT

| System | PA P | PA R | PA F1 | PA F1+ | PA F1s | SPOT P | SPOT R | SPOT F1 | SPOT F1+ | SPOT F1s |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (orchestrate-math) | .268 | **.576** | **.343** | **.631** | **.343** | **.267** | **.595** | **.344** | **.584** | **.344** |
| *— single-call LLM baselines —* | | | | | | | | | | |
| Claude-Opus-5 | .283 | .167 | .204^* | .209 | .182 | .135 | .583 | .210^* | .581 | .189 |
| Claude-Sonnet-5 | **.399** | .330 | .335 | .388 | .333 | .147 | .243 | .178 | .225 | .143 |
| DeepSeek-V4-pro | .242 | .178 | .187^* | .224 | .198 | .223 | .541 | .294 | .513 | .300 |
| GPT-5.6-terra | .316 | .333 | .289 | .383 | .298 | .210 | **.595** | .293 | .562 | .333 |

`F1+` = out-of-scope findings removed from the precision denominator; `F1s` = strict regime (baseline must itself label the finding *Equation / proof*). All systems use the same default taxonomy. Macro-averages over papers; one shared LLM judge (Claude-Sonnet-5). `*` = ours better at p<0.05 (paired Wilcoxon on per-paper F1, Holm-corrected).

