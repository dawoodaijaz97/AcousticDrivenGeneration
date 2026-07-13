# B17 — Final Flan-T5 Model: Results & Interpretation

**Status:** final model for the Flan-T5 track. **Date:** 2026-07-13.

## Executive summary

**B17 is our best Flan-T5 model for generating mFDA-style clinical speech reports from
seven acoustic biomarkers.** Evaluated on the paper's own metric, it reaches **AVG 0.649**
(char-BLEU 0.759, BERT 0.911) — within ~0.03 of the paper's best **7-billion-parameter**
systems (~0.68 AVG), from a model **28× smaller** (248M parameters).

The long-standing impression that "our Flan-T5 scores far below the paper" was **mostly a
measurement artifact, not a model deficiency**:

1. The paper reports **character-level BLEU-2**; our standard evaluation used **word-level
   BLEU-4**. For the same B17 outputs, BLEU reads **0.76** vs **0.36** depending only on which
   BLEU is used.
2. Our BERTScore defaulted to the wrong language (`es` instead of `en` — the reports are
   English). Fixing it raised B17's BERT from 0.823 to **0.911**.

Neither changes a single generated word; both were rulers, not results.

## The model

| Property | Value |
| --- | --- |
| Base | `google/flan-t5-base` (248M) |
| Adaptation | Full fine-tune 5 epochs (**B11**), then **LoRA rank 32** for 3 epochs (**B17**), adapters merged |
| Prompt | `flan-paper` — `Generate a report for:` + seven compact biomarker scores |
| Training data | 100k **synthetic** prompt→report pairs |
| Learning rate / weight decay | 5e-4 / 0.01 |
| Test set | **96 real** clinical reports (48 PD, 48 HC), held out |
| Checkpoint | `runs/flan-t5-base/100k-flan-paper-5ep-lora32/final_model` |

Input is the seven mFDA biomarkers (Breathing, Lips, Palate, Larynx, Monotonicity, Tongue,
Intelligibility) formatted as a compact score string; output is the free-text mFDA-style report.

## Results

### Honest metrics — `main.eval_decode` (standard, for model selection)

Word-level SacreBLEU-4, ROUGE (bootstrap mid-F1), **BERTScore `lang=en`** (roberta-large),
deterministic beam search (beam 3, 512 tokens).

| Group | R-1 | R-2 | R-L | BLEU (word) | BERT | **AVG** |
| --- | --- | --- | --- | --- | --- | --- |
| **Overall** | 0.605 | 0.389 | 0.530 | 0.362 | 0.911 | **0.559** |
| PD | 0.582 | 0.336 | 0.491 | 0.342 | 0.909 | 0.532 |
| HC | 0.625 | 0.441 | 0.569 | 0.383 | 0.914 | 0.586 |

*(BERT/AVG shown are the corrected `en` values; the earlier `es` default gave BERT 0.823 / AVG 0.542.)*

### Paper-comparable metrics — `main.eval_author_parity` (reproduces the authors' code)

Char-level BLEU-2 (their `nltk.sentence_bleu` on raw strings), stemmed per-example ROUGE F1,
**BERTScore `lang=en`**, their sampling decode (`do_sample=True, temperature=0.7, top_k=50,
min_length=100, beam 3`).

| Group | R-1 | R-2 | R-L | BLEU (char) | BERT | **AVG** |
| --- | --- | --- | --- | --- | --- | --- |
| **Overall** | 0.632 | 0.394 | 0.548 | **0.759** | 0.911 | **0.649** |
| PD | 0.611 | 0.344 | 0.508 | 0.744 | 0.909 | 0.623 |
| HC | 0.652 | 0.444 | 0.588 | 0.775 | 0.912 | 0.674 |

### Paper reference (from the paper's Table 4 / abstract)

| | BLEU | AVG |
| --- | --- | --- |
| Best 7B models (LLaMA-7B, Mistral-7B, DeepSeek-7B) | **0.789** (PD) / **0.836** (HC), char-level | ~**0.68** (LLaMA-7B) |

**Compare B17's parity numbers (0.649 AVG; 0.744/0.775 char-BLEU) to these, not the honest
numbers** — the paper's figures are on the char-BLEU ruler.

## Why the apparent gap was a metric artifact

- **BLEU definition.** The paper's evaluation passes raw strings to `nltk.sentence_bleu(...,
  weights=(0.5,0.5))`. nltk then tokenizes into **characters**, so their "BLEU" is char-level
  unigram+bigram. Two *different* English clinical reports already score ~0.6–0.84 this way
  because they share an alphabet and boilerplate. Standard word-level BLEU-4 is far stricter.
  Result: identical B17 outputs read as **0.36 (word)** vs **0.76 (char)**.
- **BERTScore language.** The source *speech* is Colombian Spanish, but the *reports* are
  written in **English**. Scoring English text with a Spanish/multilingual model (`lang=es`)
  under-scored BERT by ~0.09. Corrected to `en` (roberta-large): BERT **0.823 → 0.911**.
- **Cross-check that this is measurement, not model:** ROUGE-2 is nearly identical across both
  rulers (0.389 honest vs 0.394 parity) because ROUGE is word-based in both — the divergence
  lives entirely in BLEU, exactly as the char-vs-word explanation predicts.

**Bottom line:** under the exact metric and decode the paper used, a 248M Flan-T5 lands beside
the paper's 7B systems. Our model was never "far below the paper."

## What was tried and is closed (Flan-T5-base plateau)

Every axis below was tested and did not beat B17 (AVG on the honest word-BLEU ruler):

| Lever | Result |
| --- | --- |
| Learning rate | 5e-4 clear win over 3e-4 (0.522 vs 0.395) |
| Epochs 3→5 (B11) | 0.529 → 0.538 |
| LoRA rank (B14/16/17) | r8 0.538, r16 0.540, **r32 0.542 (B17, best)** |
| Longer LoRA (B18) | ties B17 |
| Weight decay | 0.01 best; 0.0 → 0.500, 0.05 → 0.517 |
| Label smoothing | degenerate (~0.176) — do not use |
| Prompt variants (B6/B7) | below baseline |
| Model size — large @ 5ep (L5) | 0.536, ties base (3× cost, no gain) |
| Non-Flan t5-base (N0) | 0.439 — Flan required |
| Severity oversampling (B20) | ties B17 |
| Structured targets (B21/B22/B23) | 0.492 / 0.531 / 0.315 — all below B17 |
| Raw `Instructions` encoder input (B24) | 0.479 — compact seven-score input wins |

Best small model for reference: **S4** (flan-t5-small), AVG 0.437.

## Persistent limitation

**PD reports are the weaker group** (honest AVG: PD 0.532 vs HC 0.586), consistent with the
paper's observation that models do better on HC. Strict seven-slot `Category (Severity):`
structure coverage in generated text is ~14% (references 100%). Closing the remaining gap to
the paper's 7B systems is a **scale/structure** problem, not a Flan-T5 tuning problem.

## Recommendation

- **For the Flan-T5 track: B17 is final.** Every tuning lever is exhausted; it is competitive
  with the paper under the paper's own metric.
- **To beat the paper's headline (0.789/0.836):** move to a **7B decoder model (LLaMA/Mistral)
  with LoRA** — the same family that produced those numbers. That is a new track, not a tweak.
- **Reporting/selection:** keep using `eval_decode` AVG (word BLEU-4) for selecting models;
  use `eval_author_parity` only for comparison against the paper.

## Reproduce

```bash
# Honest metrics (word BLEU-4, BERT en)
python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-lora32/final_model \
  --tokenizer-model google/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-lora32/test_decode_metrics.json \
  --require-gpu

# Paper-comparable metrics (char BLEU-2, sampling decode, BERT en)
python -m main.eval_author_parity \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-lora32/final_model \
  --tokenizer-model google/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-lora32/test_author_parity_metrics.json \
  --require-gpu
```

BERTScore (`en`) needs `roberta-large` cached; on the offline HPC GPU nodes, pre-cache it on
the login node first (see [hpc-commands.md](hpc-commands.md)).

## Artifacts

- `runs/flan-t5-base/100k-flan-paper-5ep-lora32/test_decode_metrics.json` — honest metrics
- `runs/flan-t5-base/100k-flan-paper-5ep-lora32/test_author_parity_metrics.json` — paper-parity metrics
- Detail & experiment log: [training_progress.md](training_progress.md) · recipe: [train.md](train.md)
