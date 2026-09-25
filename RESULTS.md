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
| Macro precision | 89.4% | 95.7% | 97.8% | **99.5%** |
| Macro recall | 90.1% | 95.5% | 98.2% | **99.5%** |
| Macro F1 | 88.5% | 95.5% | 98.0% | **99.4%** |

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

**Precision and recall by answer label**

Accuracy counts items, so it rewards getting the common answers right. Here every answer label is scored on its own: *precision* is how often the model is right when it gives that answer, *recall* is how often it gives that answer when it's the right one. The macro average weights every label equally within its family, and the overall macro averages the families equally, so a model that ignores rare answers scores lower.

| Task family | Labels | TEV (Together) P / R / F1 | JEV (AI Space) P / R / F1 | GLM 5.3 (AI Space) P / R / F1 | Claude Opus 5.5 (AI Space) P / R / F1 |
|---|---:|---:|---:|---:|---:|
| `action_review` | 3 | 87 / 88 / 84 | 100 / 100 / 100 | 98 / 99 / 98 | 100 / 100 / 100 |
| `agent_routing` | 16 | 95 / 95 / 93 | 98 / 98 / 98 | 100 / 100 / 100 | 99 / 99 / 99 |
| `claim_support` | 3 | 95 / 89 / 91 | 97 / 93 / 95 | 100 / 100 / 100 | 100 / 100 / 100 |
| `content_moderation` | 10 | 85 / 96 / 89 | 99 / 100 / 99 | 99 / 100 / 99 | 99 / 100 / 99 |
| `returns_policy` | 10 | 84 / 88 / 85 | 86 / 85 / 86 | 100 / 100 / 100 | 100 / 100 / 100 |
| `review_sentiment` | 5 | 83 / 79 / 80 | 98 / 98 / 98 | 98 / 98 / 98 | 98 / 98 / 98 |
| `support_intent` | 40 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| `ticket_triage` | 10 | 86 / 87 / 86 | 88 / 90 / 89 | 88 / 90 / 89 | 100 / 100 / 100 |

Per label, as precision / recall (%). *n* is how many items have that label as the right answer; a label with n = 0 was never right but some model picked it.

<details><summary><code>action_review</code> (3 labels)</summary>

| Label | n | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|---:|
| `auto_approve` | 24 | 100 / 62 | 100 / 100 | 100 / 96 | 100 / 100 |
| `require_human_review` | 14 | 61 / 100 | 100 / 100 | 93 / 100 | 100 / 100 |
| `block` | 12 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

</details>

<details><summary><code>agent_routing</code> (16 labels)</summary>

| Label | n | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|---:|
| `sql_query` | 6 | 100 / 100 | 100 / 100 | 100 / 100 | 86 / 100 |
| `calendar_create_event` | 5 | 83 / 100 | 100 / 100 | 100 / 100 | 100 / 80 |
| `web_search` | 5 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `ask_clarifying_question` | 4 | 100 / 25 | 100 / 100 | 100 / 100 | 100 / 100 |
| `billing_agent` | 4 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `code_interpreter` | 4 | 80 / 100 | 100 / 75 | 100 / 100 | 100 / 100 |
| `send_email` | 4 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `account_security_agent` | 3 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `knowledge_base_lookup` | 3 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `reminder_create` | 3 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `calendar_lookup` | 2 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `respond_directly` | 2 | 100 / 100 | 67 / 100 | 100 / 100 | 100 / 100 |
| `sales_agent` | 2 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `human_handoff` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `refuse` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `technical_support_agent` | 1 | 50 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

</details>

<details><summary><code>claim_support</code> (3 labels)</summary>

| Label | n | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|---:|
| `supported` | 25 | 96 / 100 | 96 / 100 | 100 / 100 | 100 / 100 |
| `contradicted` | 18 | 89 / 94 | 94 / 94 | 100 / 100 | 100 / 100 |
| `not_enough_info` | 7 | 100 / 71 | 100 / 86 | 100 / 100 | 100 / 100 |

</details>

<details><summary><code>content_moderation</code> (10 labels)</summary>

| Label | n | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|---:|
| `allow` | 25 | 95 / 72 | 100 / 96 | 100 / 96 | 100 / 96 |
| `remove_harassment` | 6 | 71 / 83 | 86 / 100 | 86 / 100 | 86 / 100 |
| `remove_prohibited_item` | 4 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `age_restrict` | 3 | 75 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `remove_rmt` | 3 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `remove_scam` | 3 | 60 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `escalate_self_harm` | 2 | 50 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `remove_spam` | 2 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `mute_spam` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `remove_underage` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

</details>

<details><summary><code>returns_policy</code> (10 labels)</summary>

| Label | n | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|---:|
| `approve_full_refund` | 23 | 100 / 78 | 88 / 96 | 100 / 100 | 100 / 100 |
| `approve_store_credit` | 7 | 78 / 100 | 75 / 86 | 100 / 100 | 100 / 100 |
| `deny_outside_window` | 7 | 64 / 100 | 100 / 71 | 100 / 100 | 100 / 100 |
| `escalate_to_human` | 4 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `deny_excluded_category` | 3 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `deny_final_sale` | 2 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `approve_exchange_only` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `approve_partial_refund` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `deny_condition` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `deny_return` | 1 | 0 / 0 | 0 / 0 | 100 / 100 | 100 / 100 |

</details>

<details><summary><code>review_sentiment</code> (5 labels)</summary>

| Label | n | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|---:|
| `very_negative` | 15 | 80 / 80 | 100 / 100 | 100 / 100 | 100 / 100 |
| `very_positive` | 13 | 81 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `positive` | 9 | 89 / 89 | 90 / 100 | 90 / 100 | 90 / 100 |
| `mixed_or_neutral` | 8 | 100 / 88 | 100 / 88 | 100 / 88 | 100 / 88 |
| `negative` | 5 | 67 / 40 | 100 / 100 | 100 / 100 | 100 / 100 |

</details>

<details><summary><code>support_intent</code> (40 labels)</summary>

| Label | n | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|---:|
| `bug_report` | 3 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `refund_request` | 3 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `billing_question` | 2 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `duplicate_charge` | 2 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `plan_change` | 2 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `setup_help` | 2 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `shipping_delay` | 2 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `warranty_claim` | 2 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `account_locked` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `add_user` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `baggage_claim` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `cancel_booking` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `cancel_order` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `cancel_service` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `cancel_subscription` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `card_declined` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `card_replacement` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `close_account` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `dispute_charge` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `downgrade_plan` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `driver_complaint` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `feature_request` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `flight_change` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `food_quality` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `late_delivery` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `loyalty_points` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `missing_item` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `open_account` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `password_reset` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `pause_subscription` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `port_number` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `praise_feedback` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `promo_code_issue` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `report_fraud` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `return_request` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `roaming_activation` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `special_assistance` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `technical_outage` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `update_address` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `wrong_order` | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

</details>

<details><summary><code>ticket_triage</code> (10 labels)</summary>

| Label | n | TEV (Together) | JEV (AI Space) | GLM 5.3 (AI Space) | Claude Opus 5.5 (AI Space) |
|---|---:|---:|---:|---:|---:|
| `sev1_outage` | 8 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `not_a_bug` | 6 | 100 / 67 | 100 / 100 | 100 / 100 | 100 / 100 |
| `payments` | 6 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `sev2_degraded` | 6 | 75 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `auth` | 5 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `frontend` | 5 | 83 / 100 | 83 / 100 | 83 / 100 | 100 / 100 |
| `infra` | 5 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `data_pipeline` | 4 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `sev3_minor` | 4 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| `other` | 1 | 0 / 0 | 0 / 0 | 0 / 0 | 100 / 100 |

</details>

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

