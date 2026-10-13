# Black Box diagnosis results

Three pipelines on the same runs: **Baseline (previous signals)** (the 25 earlier signals, model score only), **New signals, no reranker** (all 32 signals, model score only), and **Black Box model** (32 signals + decisive-evidence reranker; this is what `/diagnose` serves).

Simulated runs come from `sim_agent.py`; framework `crewai` and model `mistral` are held out entirely and train and test never share a source run. REAL runs come from Part 1's recorder (`blackbox_part 1/runs`, models: groq:openai/gpt-oss-120b, llama3, llama3:latest, mock-large, mock-small). Real runs are split by task: 164 runs on training tasks, 125 independent runs on held-out tasks. Features, reranker and threshold were chosen by cross-validation inside the training tasks; held-out tasks are only scored.

| Test set | Runs | Method | Top-1 | Top-3 | Trained on |
|---|---|---|---|---|---|
| SIMULATED, seen setup, held-out source runs | 78 | Black Box model | 97.4% | 100.0% | 324 simulated |
| SIMULATED, seen setup, held-out source runs | 78 | New signals, no reranker | 97.4% | 100.0% | 324 simulated |
| SIMULATED, seen setup, held-out source runs | 78 | Baseline (previous signals) | 97.4% | 98.7% | 324 simulated |
| SIMULATED, seen setup, held-out source runs | 78 | Heuristic (first error, else last step) | 43.6% | 60.3% | - |
| SIMULATED, unseen setup (crewai or mistral) | 498 | Black Box model | 96.4% | 98.8% | 324 simulated |
| SIMULATED, unseen setup (crewai or mistral) | 498 | New signals, no reranker | 96.4% | 98.8% | 324 simulated |
| SIMULATED, unseen setup (crewai or mistral) | 498 | Baseline (previous signals) | 96.4% | 99.2% | 324 simulated |
| SIMULATED, unseen setup (crewai or mistral) | 498 | Heuristic (first error, else last step) | 52.4% | 64.9% | - |
| REAL, every labelled run (twins removed) | 288 | Black Box model (simulated only) | 79.2% | 92.7% | 324 simulated, no real runs |
| REAL, every labelled run (twins removed) | 288 | New signals, no reranker | 53.1% | 68.4% | 324 simulated, no real runs |
| REAL, every labelled run (twins removed) | 288 | Baseline (previous signals) | 44.1% | 77.4% | 324 simulated, no real runs |
| REAL, every labelled run (twins removed) | 288 | Heuristic (first error, else last step) | 19.8% | 44.8% | - |
| REAL, seen setup, held-out tasks (twins removed) | 125 | Black Box model (simulated + 164 real) | 99.2% | 100.0% | 324 simulated + 164 real |
| REAL, seen setup, held-out tasks (twins removed) | 125 | New signals, no reranker | 96.8% | 100.0% | 324 simulated + 164 real |
| REAL, seen setup, held-out tasks (twins removed) | 125 | Baseline (previous signals) | 92.8% | 100.0% | 324 simulated + 164 real |
| REAL, seen setup, held-out tasks (twins removed) | 125 | Heuristic (first error, else last step) | 20.0% | 48.8% | - |
| REAL LLM runs only (groq:openai/gpt-oss-120b, llama3:latest), held-out tasks | 35 | Black Box model (simulated + 164 real) | 100.0% | 100.0% | 324 simulated + 164 real |
| REAL LLM runs only (groq:openai/gpt-oss-120b, llama3:latest), held-out tasks | 35 | New signals, no reranker | 100.0% | 100.0% | 324 simulated + 164 real |
| REAL LLM runs only (groq:openai/gpt-oss-120b, llama3:latest), held-out tasks | 35 | Baseline (previous signals) | 94.3% | 100.0% | 324 simulated + 164 real |
| REAL LLM runs only (groq:openai/gpt-oss-120b, llama3:latest), held-out tasks | 35 | Heuristic (first error, else last step) | 20.0% | 62.9% | - |
| REAL, unseen model: `groq:openai/gpt-oss-120b` removed from training, held-out tasks | 34 | Black Box model (simulated + 141 real) | 100.0% | 100.0% | 324 simulated + 141 real (none from this model) |
| REAL, unseen model: `groq:openai/gpt-oss-120b` removed from training, held-out tasks | 34 | New signals, no reranker | 100.0% | 100.0% | 324 simulated + 141 real (none from this model) |
| REAL, unseen model: `groq:openai/gpt-oss-120b` removed from training, held-out tasks | 34 | Baseline (previous signals) | 82.4% | 97.1% | 324 simulated + 141 real (none from this model) |
| REAL, unseen model: `groq:openai/gpt-oss-120b` removed from training, held-out tasks | 34 | Heuristic (first error, else last step) | 17.6% | 61.8% | - |
| REAL, unseen model: `llama3:latest` removed from training, held-out tasks | 1 | Black Box model (simulated + 164 real) | 100.0% | 100.0% | 324 simulated + 164 real (none from this model) |
| REAL, unseen model: `llama3:latest` removed from training, held-out tasks | 1 | New signals, no reranker | 100.0% | 100.0% | 324 simulated + 164 real (none from this model) |
| REAL, unseen model: `llama3:latest` removed from training, held-out tasks | 1 | Baseline (previous signals) | 100.0% | 100.0% | 324 simulated + 164 real (none from this model) |
| REAL, unseen model: `llama3:latest` removed from training, held-out tasks | 1 | Heuristic (first error, else last step) | 100.0% | 100.0% | - |
| REAL, twin runs (LangGraph copies of runs above, not independent) | 91 | Black Box model (simulated + 164 real) | 98.9% | 100.0% | 324 simulated + 164 real |

## Validation (how the choices were made)

Grouped 5-fold cross-validation over the 164 real training runs: each fold holds out whole tasks, the model is trained on the simulated runs plus the other folds. 164 out-of-fold runs.

| Pipeline | Runs | Top-1 | Top-3 |
|---|---|---|---|
| Baseline (previous signals) | 164 | 91.5% | 98.2% |
| New signals, no reranker | 164 | 97.6% | 99.4% |
| Black Box model | 164 | 99.4% | 100.0% |

Flag threshold, on the same out-of-fold scores (a step is flagged when its score reaches the threshold):

| Threshold | Baseline P | Baseline R | Baseline F1 | Final P | Final R | Final F1 |
|---|---|---|---|---|---|---|
| 0.3 | 64.2% | 93.9% | 76.2% | 91.6% | 100.0% | 95.6% |
| 0.5 (old) | 71.2% | 92.1% | 80.3% | 95.9% | 100.0% | 97.9% |
| 0.6 | 73.6% | 90.2% | 81.1% | 96.5% | 100.0% | 98.2% |
| 0.7 | 77.8% | 89.6% | 83.3% | 98.8% | 100.0% | 99.4% |
| 0.8 (chosen) | 82.5% | 86.0% | 84.2% | 98.8% | 100.0% | 99.4% |
| 0.9 | 91.9% | 75.6% | 82.9% | 99.4% | 99.4% | 99.4% |
| 0.95 | 94.4% | 71.3% | 81.2% | 99.4% | 97.0% | 98.1% |

Chosen threshold: 0.8. The final pipeline's F1 is flat between 0.7 and 0.9 on validation, so the middle of that range was taken instead of the single best value.

## Precision, recall, F1 and confusion matrix (per step, held-out data)

Every step is one decision. Each run has exactly one truly faulty step.

| Test set | Steps | Precision | Recall | F1 | Pipeline | Threshold |
|---|---|---|---|---|---|---|
| SIMULATED, unseen setup | 5800 | 97.6% | 91.8% | 94.6% | Black Box model | 0.8 |
| SIMULATED, unseen setup | 5800 | 88.5% | 94.2% | 91.2% | Black Box model | 0.5 |
| SIMULATED, unseen setup | 5800 | 92.3% | 94.0% | 93.1% | Baseline (previous signals) | 0.5 |
| REAL, held-out tasks | 825 | 99.2% | 100.0% | 99.6% | Black Box model | 0.8 |
| REAL, held-out tasks | 825 | 98.4% | 100.0% | 99.2% | Black Box model | 0.5 |
| REAL, held-out tasks | 825 | 67.3% | 90.4% | 77.1% | Baseline (previous signals) | 0.5 |
| REAL LLM runs only, held-out tasks | 221 | 100.0% | 100.0% | 100.0% | Black Box model | 0.8 |
| REAL LLM runs only, held-out tasks | 221 | 100.0% | 100.0% | 100.0% | Black Box model | 0.5 |
| REAL LLM runs only, held-out tasks | 221 | 70.0% | 80.0% | 74.7% | Baseline (previous signals) | 0.5 |

Confusion matrix: SIMULATED, unseen setup, Black Box model, threshold 0.8

| | Flagged faulty | Not flagged |
|---|---|---|
| Truly faulty step | 457 (TP) | 41 (FN) |
| Healthy step | 11 (FP) | 5291 (TN) |

Confusion matrix: SIMULATED, unseen setup, Baseline (previous signals), threshold 0.5

| | Flagged faulty | Not flagged |
|---|---|---|
| Truly faulty step | 468 (TP) | 30 (FN) |
| Healthy step | 39 (FP) | 5263 (TN) |

Confusion matrix: REAL, held-out tasks, Black Box model, threshold 0.8

| | Flagged faulty | Not flagged |
|---|---|---|
| Truly faulty step | 125 (TP) | 0 (FN) |
| Healthy step | 1 (FP) | 699 (TN) |

Confusion matrix: REAL, held-out tasks, Baseline (previous signals), threshold 0.5

| | Flagged faulty | Not flagged |
|---|---|---|
| Truly faulty step | 113 (TP) | 12 (FN) |
| Healthy step | 55 (FP) | 645 (TN) |

Confusion matrix: REAL LLM runs only, held-out tasks, Black Box model, threshold 0.8

| | Flagged faulty | Not flagged |
|---|---|---|
| Truly faulty step | 35 (TP) | 0 (FN) |
| Healthy step | 0 (FP) | 186 (TN) |

Confusion matrix: REAL LLM runs only, held-out tasks, Baseline (previous signals), threshold 0.5

| | Flagged faulty | Not flagged |
|---|---|---|
| Truly faulty step | 28 (TP) | 7 (FN) |
| Healthy step | 12 (FP) | 174 (TN) |

## By fault type

### SIMULATED unseen setup (498 runs, trained on 324 simulated)

| Fault | Runs | Baseline top-1 | Baseline top-3 | Final top-1 | Final top-3 |
|---|---|---|---|---|---|
| empty_search | 93 | 100.0% | 100.0% | 100.0% | 100.0% |
| skipped_step | 103 | 100.0% | 100.0% | 100.0% | 100.0% |
| wrong_number | 106 | 100.0% | 100.0% | 100.0% | 100.0% |
| wrong_tool | 108 | 99.1% | 100.0% | 100.0% | 100.0% |
| wrong_tool_result | 88 | 80.7% | 95.5% | 79.5% | 93.2% |
| all | 498 | 96.4% | 99.2% | 96.4% | 98.8% |

### REAL held-out tasks (125 runs, trained on 324 simulated + 164 real)

| Fault | Runs | Baseline top-1 | Baseline top-3 | New signals top-1 | New signals top-3 | Final top-1 | Final top-3 |
|---|---|---|---|---|---|---|---|
| empty_search | 25 | 96.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| skipped_step | 24 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| wrong_number | 25 | 84.0% | 100.0% | 88.0% | 100.0% | 100.0% | 100.0% |
| wrong_tool | 25 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| wrong_tool_result | 26 | 84.6% | 100.0% | 96.2% | 100.0% | 96.2% | 100.0% |
| all | 125 | 92.8% | 100.0% | 96.8% | 100.0% | 99.2% | 100.0% |

### REAL LLM runs only, held-out tasks (35 runs, same training)

| Fault | Runs | Baseline top-1 | Baseline top-3 | Final top-1 | Final top-3 |
|---|---|---|---|---|---|
| empty_search | 7 | 100.0% | 100.0% | 100.0% | 100.0% |
| skipped_step | 6 | 100.0% | 100.0% | 100.0% | 100.0% |
| wrong_number | 7 | 100.0% | 100.0% | 100.0% | 100.0% |
| wrong_tool | 7 | 100.0% | 100.0% | 100.0% | 100.0% |
| wrong_tool_result | 8 | 75.0% | 100.0% | 100.0% | 100.0% |
| all | 35 | 94.3% | 100.0% | 100.0% | 100.0% |

### REAL held-out tasks, simulated-only training (125 runs, trained on 324 simulated, no real runs)

| Fault | Runs | Baseline top-1 | Baseline top-3 | New signals top-1 | New signals top-3 | Final top-1 | Final top-3 |
|---|---|---|---|---|---|---|---|
| empty_search | 25 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| skipped_step | 24 | 25.0% | 50.0% | 54.2% | 58.3% | 54.2% | 58.3% |
| wrong_number | 25 | 68.0% | 68.0% | 56.0% | 68.0% | 88.0% | 100.0% |
| wrong_tool | 25 | 0.0% | 64.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| wrong_tool_result | 26 | 30.8% | 100.0% | 73.1% | 100.0% | 73.1% | 100.0% |
| all | 125 | 44.8% | 76.8% | 56.8% | 65.6% | 83.2% | 92.0% |

## Where the top suspect lands (final pipeline)

### SIMULATED unseen setup

Rows: kind of the truly faulty step. Columns: what was ranked first.

| Faulty step is a | right step | wrong step, model_call | wrong step, tool_call | wrong step, decision |
|---|---|---|---|---|
| model_call | 209 | 0 | 0 | 0 |
| tool_call | 254 | 0 | 17 | 1 |
| decision | 17 | 0 | 0 | 0 |

### REAL held-out tasks

Rows: kind of the truly faulty step. Columns: what was ranked first.

| Faulty step is a | right step | wrong step, model_call | wrong step, tool_call | wrong step, decision |
|---|---|---|---|---|
| model_call | 17 | 0 | 0 | 0 |
| tool_call | 82 | 1 | 0 | 0 |
| decision | 25 | 0 | 0 | 0 |

## Unseen failure types (final pipeline)

For each row the model is retrained with that fault type removed from training, then tested only on runs with that fault. "Seen" is the normal model, which had examples of it.

| Fault type held out | Test set | Runs | Top-1 never seen | Top-3 never seen | Top-1 seen | Top-3 seen |
|---|---|---|---|---|---|---|
| empty_search | SIMULATED, unseen setup | 93 | 86.0% | 100.0% | 100.0% | 100.0% |
| empty_search | REAL, held-out tasks | 25 | 0.0% | 96.0% | 100.0% | 100.0% |
| skipped_step | SIMULATED, unseen setup | 103 | 100.0% | 100.0% | 100.0% | 100.0% |
| skipped_step | REAL, held-out tasks | 24 | 79.2% | 79.2% | 100.0% | 100.0% |
| wrong_number | SIMULATED, unseen setup | 106 | 19.8% | 24.5% | 99.1% | 100.0% |
| wrong_number | REAL, held-out tasks | 25 | 32.0% | 44.0% | 100.0% | 100.0% |
| wrong_tool | SIMULATED, unseen setup | 108 | 84.3% | 86.1% | 99.1% | 100.0% |
| wrong_tool | REAL, held-out tasks | 25 | 100.0% | 100.0% | 100.0% | 100.0% |
| wrong_tool_result | SIMULATED, unseen setup | 88 | 21.6% | 50.0% | 73.9% | 96.6% |
| wrong_tool_result | REAL, held-out tasks | 26 | 15.4% | 96.2% | 96.2% | 100.0% |
| **all, each held out in turn** | SIMULATED, unseen setup | 498 | 63.1% | 72.1% | 95.0% | 99.4% |
| **all, each held out in turn** | REAL, held-out tasks | 125 | 44.8% | 83.2% | 99.2% | 100.0% |

## How often each signal fires

On faulty steps / on healthy steps. Real numbers are from the training tasks only.

| Signal | Simulated faulty | Simulated healthy | Real faulty | Real healthy |
|---|---|---|---|---|
| tool_error_never_retried | 37% | 0% | 19% | 4% |
| tool_error | 37% | 3% | 19% | 4% |
| calculation_does_not_match_its_input | 0% | 0% | 5% | 0% |
| wrote_a_request_a_tool_rejected | 0% | 0% | 21% | 0% |
| tool_result_is_about_something_else | 10% | 7% | 20% | 0% |
| steps_that_used_it_got_nothing | 3% | 5% | 55% | 16% |
| first_step_to_go_wrong | 33% | 2% | 38% | 4% |
| no_usable_result | 0% | 0% | 19% | 12% |
| empty_result | 37% | 3% | 20% | 0% |
| output_contradicts_earlier_steps | 42% | 4% | 23% | 9% |
| input_not_backed_by_earlier_steps | 0% | 3% | 21% | 13% |
| next_step_failed_because_of_it | 3% | 5% | 21% | 4% |
| repeated_action | 3% | 2% | 0% | 0% |
| result_ignored_by_later_steps | 0% | 4% | 0% | 0% |

## Learning from confirmed runs (final pipeline)

74 runs confirmed by the verify loop were added to a simulated-only model, labelled with the step replay confirmed (not the injected label). Tested on held-out runs (simulated unseen setup, plus real held-out runs when given) whose task or source run is not among the added ones.

| Runs | Trained on | Top-1 | Top-3 |
|---|---|---|---|
| 564 | 324 simulated | 95.0% | 98.2% |
| 564 | 324 simulated + 74 replay-confirmed | 94.1% | 98.9% |
