# JEV vs TEV

[Jev](https://flaviocopes.com/jev/) is TypeSafe's closed-source decision model. Together AI [trained a Jev-style model for $17](https://www.together.ai/blog/how-to-train-your-own-jev), `Tev1-4B-experimental` ([announcement](https://www.linkedin.com/feed/update/urn:li:activity:7508652815490105344)), but published no accuracy numbers. We tested it: same 400 new tasks for both models, with two frontier LLMs alongside for reference.

![Cost vs accuracy: TEV and JEV cost about 1–2 cents per 1,000 tasks; GLM 5.3 and Opus 5.5 score about 2 points higher at 41–87× the price](docs/img/cost-vs-accuracy.png)

## Results

<!-- RESULTS:START -->
**JEV (AI Space)** is more accurate by 7.3 points; **TEV (Together)** is 2.7× faster at p50; **TEV (Together)** is 1.9× cheaper per task.

| | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|
| Cost per task | **$0.0000103** | $0.0000196 | $0.0008097 | $0.0017083 |
| Speed (p50) | **173 ms** | 459 ms | 2,879 ms | 2,477 ms |
| Accuracy | 90.0% | 97.2% | 99.0% | **99.2%** |
| Both halves of a pair right | 80.0% | 95.0% | 98.0% | **98.5%** |
| Macro F1 (every answer weighted equally) | 88.5% | 95.5% | 98.0% | **99.4%** |

400 held-out items (200 contrastive pairs, 8 task families). The whole experiment cost **$1.08** in API calls. Per-label precision and recall, per-family scores, prompt variants, calibration and significance: [RESULTS.md](RESULTS.md).
<!-- RESULTS:END -->

![Median latency per call: TEV 173 ms, JEV 459 ms, Opus 5.5 2,477 ms, GLM 5.3 2,879 ms](docs/img/speed.png)

## How it works

![Two halves of a contrastive pair differ by one word; both go through the same prompt to four models, which are scored on accuracy, pair accuracy, cost and speed](docs/img/how-it-works.png)

The 400 tasks are new, and none come from the public datasets TEV was trained on. Details and caveats: [METHODOLOGY.md](METHODOLOGY.md).

## Prompts matter too

![Accuracy change from the default prompt: rewording moves TEV and JEV by under a point, while removing option descriptions costs TEV 6.8 points and JEV 4.5](docs/img/prompts-matter.png)

## Reproduce

```bash
cp .env.example .env    # add TOGETHER_API_KEY and AISPACE_API_KEY
uv sync
uv run python -m bench.run --provider tev jev glm opus
uv run python -m bench.report   # README table + RESULTS.md
uv run python -m bench.charts   # README diagrams (needs Google Chrome)
uv run python -m bench.spend    # what you've spent
```
