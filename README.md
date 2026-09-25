# JEV vs TEV

[Jev](https://flaviocopes.com/jev/) is TypeSafe's closed-source decision model. Together AI [trained a Jev-style model for $17](https://www.together.ai/blog/how-to-train-your-own-jev), `Tev1-4B-experimental` ([announcement](https://www.linkedin.com/feed/update/urn:li:activity:7508652815490105344)), but published no accuracy numbers. We tested it: same 400 new tasks for both models, with two frontier LLMs alongside for reference.

## Results

<!-- RESULTS:START -->
**JEV (AI Space)** is more accurate by 7.3 points; **TEV (Together)** is 2.7× faster at p50; **TEV (Together)** is 1.9× cheaper per task.

| | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|
| Cost per task | **$0.0000103** | $0.0000196 | $0.0008097 | $0.0017083 |
| Speed (p50) | **173 ms** | 459 ms | 2,879 ms | 2,477 ms |
| Accuracy | 90.0% | 97.2% | 99.0% | **99.2%** |
| Both halves of a pair right | 80.0% | 95.0% | 98.0% | **98.5%** |

### Prompts matter too

Same 400 items, same models, five prompt versions. Δ is the change from the default prompt; *answers changed* counts items where the answer differs from the default prompt's answer.

| Prompt | TEV accuracy | JEV accuracy | TEV answers changed | JEV answers changed |
|---|---:|---:|---:|---:|
| `default`: vendor-recommended prompt | 90.0% | 97.2% | – | – |
| `careful`: + one line of reading guidance | 89.8% (-0.3) | 97.2% (+0.0) | 1.2% | 1.0% |
| `reversed`: same options, reverse order | 90.8% (+0.7) | 97.2% (+0.0) | 4.2% | 1.0% |
| `generic_question`: "Which option best fits the input?" | 89.5% (-0.5) | 97.0% (-0.3) | 4.0% | 1.8% |
| `keys_only`: option keys, no descriptions | 83.2% (-6.8) | 92.8% (-4.5) | 12.2% | 5.5% |

Best vs worst prompt: TEV 7.5 points, JEV 4.5 points. Biggest single effect: `keys_only`. Answers can change even when accuracy doesn't: a prompt can fix some items and break others.

400 held-out items (200 contrastive pairs, 8 task families). The whole experiment cost **$1.08** in API calls. Per-family scores, prompt variants, calibration and significance: [RESULTS.md](RESULTS.md).
<!-- RESULTS:END -->

## How it works

- **Tasks:** 400 new items across 8 kinds of decisions (support intent, return policy, agent routing, moderation, sentiment, claim checking, action approval, ticket triage). None come from the public datasets TEV was trained on.
- **Contrastive pairs:** items come in pairs where a small edit flips the right answer (staging → prod, day 30 → day 31). "Both halves right" catches models that match keywords without reading closely.
- **Same input for every model:** TEV, GLM 5.3 and Opus 5.5 get Together's recommended prompt. JEV gets the same text as a `choice` question.
- **Cost per task** = billed tokens × list price. **Speed** = wall-clock time per call as seen from the client.

Details and caveats: [METHODOLOGY.md](METHODOLOGY.md).

## Reproduce

```bash
cp .env.example .env    # add TOGETHER_API_KEY and AISPACE_API_KEY
uv sync
uv run python -m bench.run --provider tev jev glm opus
uv run python -m bench.report   # updates this README and RESULTS.md
uv run python -m bench.spend    # what you've spent
```
