# JEV vs TEV: a head-to-head decision-model benchmark

## Why this exists

[Jev](https://flaviocopes.com/jev/) is TypeSafe's closed-source "System One" model. It doesn't generate text. You give it an input and a typed question, and it returns a decision with probabilities over the options. TypeSafe hasn't published the weights or the training recipe. You can only call it through the API.

Shortly after Jev launched, Together AI [published a recipe for training your own Jev for $17](https://www.together.ai/blog/how-to-train-your-own-jev) ([announcement](https://www.linkedin.com/feed/update/urn:li:activity:7508652815490105344)). They fine-tuned Qwen3.5 4B on about 38k public classification examples in about 25 minutes, and serve the result as `together/Tev1-4B-experimental` ("TEV"). The post reports what training cost and how long it took. It reports no accuracy numbers, and it doesn't compare TEV with Jev.

So we tested whether a $17 open fine-tune can do Jev's job. Both models get the same held-out decision tasks: data neither was trained on, written as contrastive pairs so that surface keyword matching doesn't pay off. We report accuracy, robustness, latency, cost and calibration side by side.

## Results

<!-- RESULTS:START -->
_No run yet. See [Reproducing](#reproducing)._
<!-- RESULTS:END -->

## What's being measured

| Model | Where it runs | How it's called |
|---|---|---|
| **TEV** `together/Tev1-4B-experimental` | Together AI serverless | Chat completions. We use the system prompt and settings from Together's launch post (`temperature=0`, `max_tokens=8`, thinking off). The item goes in as the post's JSON (state, question, lettered options), and the model replies with one letter. We read letter probabilities from the first token's logprobs. |
| **JEV** `jev-latest` | TypeSafe via AI Space (`/v1/systemone`) | One `choice` question per item. Each option key maps to its description in `criteria`. Jev returns the choice and a probability for each option. |

Both models see the same text for every item: the same state, question, option keys and descriptions, in the same order.

### Dataset: contrastive pairs

The dataset is in [`data/`](data/), and [`data/SPEC.md`](data/SPEC.md) describes the format. It has **400 items across 8 task families**. Items come in **200 contrastive pairs**. The two halves of a pair share the question and options. Their inputs differ by a small, realistic edit that changes the right answer: a negation, a date one day past a return window, `--dry-run` removed from a command, an internal recipient swapped for an external one. So we can report two numbers:

- **Accuracy**: the share of items answered correctly.
- **Pair accuracy**: the share of pairs where *both* halves are right. A model that matches on topic words and doesn't read closely gets one half right and the other wrong, so this number drops.

| Task family | What it tests |
|---|---|
| `support_intent` | Customer message → support intent, including multi-issue messages and negated intents |
| `returns_policy` | Apply a written return policy (day windows, exceptions, final-sale rules) to a request |
| `agent_routing` | Route an agent request to the right tool, or ask a clarifying question |
| `content_moderation` | Apply a platform policy to a post: quoting abuse vs committing it, look-alike items |
| `review_sentiment` | 5-level sentiment with "but" clauses, sarcasm and intensity shifts |
| `claim_support` | Evidence + claim → supported / contradicted / not enough info |
| `action_review` | Approve, send for human review, or block an agent's proposed shell/SQL/API/email action |
| `ticket_triage` | Bug and incident reports → severity or owning team |

Claude wrote all items for this benchmark. They are original text, not taken from Banking77, AG News, SST-5, BoolQ or MultiNLI. TEV was fine-tuned on samples from those datasets, so reusing them would test memorisation more than generalisation. About 40% of pairs are marked `hard`.

### Metrics

- **Accuracy, 95% CI**: bootstrap that resamples whole pairs, since the two halves of a pair aren't independent.
- **Head to head**: exact McNemar test on the items where exactly one model is right.
- **Unusable output**: replies that don't map to an option (TEV), or errors after 5 retries. These count as wrong.
- **Latency**: client-side wall-clock time per request, p50 and p95. Both models run from the same machine, one after the other, 4 requests in flight, with 3 unrecorded warm-up calls. This includes network time to each provider, so it measures what a caller sees, not the model alone.
- **Cost per 1k decisions**: average tokens each API reports × list price from `.env`.
- **Calibration**: ECE (10 bins) and Brier score, computed on each model's probability for its top choice. TEV's probabilities come from first-token logprobs, renormalised over the option letters.

### Caveats

- A language model wrote the data, and the same kind of model checked the labels. Labels were validated structurally and spot-checked. A careful human could still dispute an item or two.
- Labels aren't balanced within every family. In `claim_support`, `content_moderation` and `action_review`, always picking the most common answer scores about 50%. The per-family table shows this majority baseline next to each model.
- The dataset is small (400 items), so category-level numbers have wide intervals. Rely on the headline CI and the McNemar p-value.
- Latency depends on region and provider load at run time.
- Only `choice` questions are tested. Jev's `score` and `noul` types have no TEV equivalent here.

## Reproducing

```bash
cp .env.example .env            # add TOGETHER_API_KEY and AISPACE_API_KEY
uv sync
uv run python -m bench.dataset  # validate the dataset
uv run python -m bench.run --provider tev jev --limit 20   # smoke test
uv run python -m bench.run --provider tev jev              # full run (resumable)
uv run python -m bench.report   # rewrite the Results section above
```

Raw predictions, including each API's raw reply, are written to `results/<provider>.jsonl`.
To try the pipeline without API keys, run `uv run python -m bench.run --provider oracle` and then `uv run python -m bench.report --provider oracle --stdout`.

Tests: `uv run pytest`.
