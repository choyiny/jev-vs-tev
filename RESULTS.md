# Results

Full numbers for the [JEV vs TEV benchmark](README.md). How they're measured: [METHODOLOGY.md](METHODOLOGY.md).

_Run on 2026-09-25 · 400 items / 200 contrastive pairs · 8 task families._

**JEV (AI Space)** is more accurate by 7.3 points; **TEV (Together)** is 2.7× faster at p50; **TEV (Together)** is 1.9× cheaper per task.

For reference, GLM 5.3 (AI Space) scores 99.0% at 2879 ms p50, $0.0008097 per task.
For reference, Claude Opus 5.5 (AI Space) scores 99.2% at 2477 ms p50, $0.0017083 per task.

| Cost · Speed · Accuracy | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|
| Cost per task | **$0.0000103** | $0.0000196 | $0.0008097 | $0.0017083 |
| Cost per 1M tasks | **$10.33** | $19.64 | $809.69 | $1,708.34 |
| Speed: latency p50 | **173 ms** | 459 ms | 2879 ms | 2477 ms |
| Speed: latency p95 | **214 ms** | 950 ms | 6705 ms | 8167 ms |
| Accuracy | 90.0% | 97.2% | 99.0% | **99.2%** |
| Accuracy 95% CI | 87.0%–92.5% | 95.5%–98.8% | 98.0%–99.8% | 98.2%–100.0% |
| Pair accuracy (both halves right) | 80.0% | 95.0% | 98.0% | **98.5%** |

**Details**

| Metric | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|
| Model version served | `together/Tev1-4B-experimental` | `jev-1.13.0` | `@cf/zai-org/glm-5.3` | `claude-opus-5-5` |
| Billed tokens per task, in / out | 246 / 2.0 | 468 / 57.8 | 240 / 107.8 | 364 / 12.7 |
| Unusable output | 0.0% | 0.0% | 0.0% | 0.0% |
| Calibration ECE ↓ | 0.031 | 0.020 | n/a | n/a |
| Brier score ↓ | 0.158 | 0.041 | n/a | n/a |

**Head to head.** Both right on 356, both wrong on 7. TEV (Together) alone right on 4; JEV (AI Space) alone right on 33. Exact McNemar p = 1.08e-06 (significant at 0.05).

**Accuracy by task family**

| Task family | n | Majority baseline | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|---:|---:|
| `action_review` | 50 | 48.0% | 82.0% | 100.0% | 98.0% | 100.0% |
| `agent_routing` | 50 | 12.0% | 94.0% | 98.0% | **100.0%** | 98.0% |
| `claim_support` | 50 | 50.0% | 94.0% | 96.0% | 100.0% | 100.0% |
| `content_moderation` | 50 | 50.0% | 84.0% | 98.0% | 98.0% | 98.0% |
| `returns_policy` | 50 | 46.0% | 88.0% | 90.0% | 100.0% | 100.0% |
| `review_sentiment` | 50 | 30.0% | 84.0% | 98.0% | 98.0% | 98.0% |
| `support_intent` | 50 | 6.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `ticket_triage` | 50 | 16.0% | 94.0% | 98.0% | 98.0% | **100.0%** |

**Accuracy by difficulty**

| Difficulty | n | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|---:|
| easy | 230 | 94.3% | 98.3% | 98.7% | 99.6% |
| hard | 170 | 84.1% | 95.9% | 99.4% | 98.8% |

**Prompt sensitivity.** The same 400 items run with three prompt versions for TEV and JEV. Δ is the accuracy change from each model's default prompt.

| Prompt | TEV (Together) accuracy | TEV (Together) Δ | TEV (Together) pair accuracy | JEV (AI Space) accuracy | JEV (AI Space) Δ | JEV (AI Space) pair accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `default`: vendor-recommended prompt | 90.0% | – | 80.0% | 97.2% | – | 95.0% |
| `careful`: + one line of reading guidance | 89.8% | -0.3 | 79.5% | 97.2% | +0.0 | 94.5% |
| `keys_only`: option keys, no descriptions | 83.2% | -6.8 | 68.0% | 92.8% | -4.5 | 86.0% |
| `reversed`: same options, reverse order | 90.8% | +0.7 | 81.5% | 97.2% | +0.0 | 94.5% |
| `generic_question`: "Which option best fits the input?" | 89.5% | -0.5 | 79.5% | 97.0% | -0.3 | 94.0% |

**What this experiment cost.** Every billed API call, including prompt variants, warm-ups and retries, priced at list rates. Pre-ledger calls that left no result row are estimated from average tokens per call.

| Model | Billed calls | Input tokens | Output tokens | USD |
|---|---:|---:|---:|---:|
| TEV (Together) | 2,020 | 471,446 | 4,040 | $0.0198 (≈$0.0001 estimated) |
| JEV (AI Space) | 2,043 | 938,528 | 118,074 | $0.0394 (≈$0.0007 estimated) |
| GLM 5.3 (AI Space) | 405 | 97,071 | 43,670 | $0.3280 (≈$0.0042 estimated) |
| Claude Opus 5.5 (AI Space) | 406 | 147,593 | 5,163 | $0.6936 (≈$0.0103 estimated) |
| **Total** | 4,874 | 1,654,638 | 170,947 | **$1.08** |

