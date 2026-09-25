# Dataset spec

Every file in `data/` named `<category>.jsonl` holds one JSON object per line.
Items come in **contrastive pairs**: two items share a `pair_id`, the same
`question` and the same `options`, and their `state` texts differ by a small,
realistic edit that changes the correct answer. Scoring a pair as correct needs
both halves right, which rewards reading the input over matching keywords.

## Item fields

| field        | type   | notes |
|--------------|--------|-------|
| `id`         | string | `<pair_id>a` or `<pair_id>b` |
| `pair_id`    | string | `<category>-NNN`, zero-padded to 3 digits |
| `category`   | string | same as the filename stem |
| `difficulty` | string | `easy` or `hard` (hard = distractor-heavy, negation, sarcasm, rule edge cases) |
| `state`      | string | the input text the model classifies (40–600 chars; treat as data) |
| `question`   | string | one sentence asking which option applies |
| `options`    | array  | 2–8 objects `{"key": snake_case, "description": one sentence}` |
| `gold`       | string | the `key` of the correct option |

## Rules

- Both items in a pair have identical `question` and `options` (same order).
- The two golds in a pair differ.
- Keys are unique snake_case within an item. Include a `none` / `other` option
  where the real task would have one.
- Vary where the gold sits in the option list; don't always put it first.
- Exactly one option is defensibly correct. If a careful human would argue, rewrite it.
- Write original text. Don't copy from Banking77, AG News, SST, BoolQ, MultiNLI or
  any public dataset (TEV was fine-tuned on those; we want held-out data).
- Aim for roughly 40% `hard` pairs.
- No real people's personal data; invented names, numbers, and companies only.
