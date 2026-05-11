# Decode evaluation — `main/eval_decode.py`

[`main/eval_decode.py`](../main/eval_decode.py) loads a **fine-tuned** seq2seq checkpoint, runs **batched greedy/beam decoding** on the **`test`** split of a tokenized `DatasetDict`, and reports **ROUGE-1 / ROUGE-2 / ROUGE-L**, **corpus BLEU**, and **BERTScore** in the same spirit as **Table 4** in [`docs/paper.pdf`](paper.pdf). It complements [`main/train.py`](train.md), which only logs teacher-forcing **loss** on `test`.

For data preparation and tokenization, see [data-pipeline.md](data-pipeline.md) and [train.md](train.md).

## Prerequisites

- Conda env **`AcousticDrivenGeneration`** (or equivalent) with **`torch`**, **`transformers`**, **`datasets`**, **`evaluate`**, **`sacrebleu`**, **`rouge-score`**, **`bert-score`**, and **`nltk`** (see [`requirements.txt`](../requirements.txt)).
- A tokenized **`DatasetDict`** that includes a **`test`** split (real mFDA reports), produced with `python -m main.prepare ... --tokenize`.
- **Same `--tokenized-dir`** (and tokenizer family) as training so encoder inputs and label decoding stay consistent.

## What it does

1. Resolves `--tokenized-dir` to a directory containing **`dataset_dict.json`**, or to `<dir>/tokenized/` if that subfolder holds the `DatasetDict`.
2. Loads **`test`** and requires **`input_ids`**, **`attention_mask`**, **`labels`**.
3. Loads model weights from **`--model-path`** (e.g. `.../checkpoint-750` or `.../final_model`) and the tokenizer from **`--tokenizer-model`** (HF id; checkpoints often omit tokenizer files).
4. Generates hypotheses with `model.generate(...)`.
5. Rebuilds **reference strings** by decoding **`labels`** with the tokenizer (padding **`-100`** stripped).
6. Computes metrics via Hugging Face **`evaluate`** (`rouge`, `sacrebleu`, `bertscore`).
7. If the **`group`** column exists (`PD` / `HC`), repeats metrics under **`by_group`**.

## What the metrics mean

These scores compare each **generated report** (hypothesis) to the **reference report** for the same example (decoded from `labels`). They are **automatic** proxies for quality; they do **not** replace clinician review. All main scores are on a **0–1** scale (higher is better), consistent with Table 4 in the paper.

### R-1, R-2, R-L (ROUGE)

**ROUGE** (*Recall-Oriented Understudy for Gisting Evaluation*) measures overlap between prediction and reference at different granularities:

| JSON key | Name | Idea |
| --- | --- | --- |
| **`R_1`** | ROUGE-1 | Overlap of **single words** (unigrams). Rises when the model uses many of the same words as the reference. |
| **`R_2`** | ROUGE-2 | Overlap of **adjacent word pairs** (bigrams). Harder to match by chance; rewards phrasing that follows the reference more closely. |
| **`R_L`** | ROUGE-L | Based on the **longest common subsequence** of words — captures shared wording and order without requiring every bigram to match. |

ROUGE is common for **summarization and report-style** text. Here it is computed via Hugging Face **`evaluate`** (bootstrap aggregate F1; see `metrics_config` in the JSON).

### BLEU

**BLEU** was developed for machine translation. It rewards **n-gram** matches (short runs of words) between prediction and reference and applies a **brevity penalty** if the output is much shorter than the reference. It can be **strict** on exact wording; good paraphrases may score lower than ROUGE. This repo uses **SacreBLEU** through **`evaluate`**; **`BLEU`** in the JSON is the corpus score on **0–1** (SacreBLEU’s 0–100 scale divided by 100).

### BERT (BERTScore)

**`BERT`** is **not** “the BERT classifier’s accuracy.” It is **BERTScore**: prediction and reference are embedded with a **multilingual BERT** (for `lang='es'`), tokens are aligned by **cosine similarity** of vectors, and scores are aggregated to precision / recall / **F1**. The script reports the **mean F1 over test examples** — it gives **partial credit for paraphrases** and similar meaning when word overlap (ROUGE/BLEU) is low.

### AVG

**`AVG`** is the **simple average** of **`R_1`**, **`R_2`**, **`R_L`**, **`BLEU`**, and **`BERT`** — the same combined summary style as the **AVG** column in the paper’s Table 4. It is not an extra independent metric.

### How to read them together

- **ROUGE** and **BLEU** emphasize **lexical overlap** and **n-gram** similarity to the reference.
- **BERTScore** adds a **semantic / embedding** view (similar meaning, different surface form).
- For **structured mFDA-style** reports (fixed categories and severity labels), high overlap metrics often mean the model matched the reference’s **structure and wording**; BERTScore still helps when wording differs but clinical meaning is close.

## Generation defaults (paper Section 3)

The paper states inference used **512 output tokens**, **beam search 3**, and **no repeating n-gram 2**. The script defaults match that:

| Setting | CLI flag | Default |
| --- | --- | --- |
| Max new tokens | `--max-new-tokens` | `512` |
| Beam width | `--num-beams` | `3` |
| No-repeat n-gram | `--no-repeat-ngram-size` | `2` (`0` = off) |

## CLI reference

| Argument | Required | Description |
| --- | --- | --- |
| `--tokenized-dir` | yes | `DatasetDict` root (or prepare output containing `tokenized/`). |
| `--model-path` | yes | Directory with fine-tuned weights (`config.json` + `model.safetensors` or `pytorch_model.bin`). |
| `--tokenizer-model` | yes | HF model id for `AutoTokenizer` (e.g. `google/flan-t5-small`). |
| `--output-json` | no | If set, writes the same JSON blob printed to stdout. |
| `--batch-size` | no | Decode batch size (default `8`). |
| `--seed` | no | PyTorch seed (default `42`). |
| `--cpu-only` | no | Force CPU. |
| `--fp16` | no | `float16` weights on CUDA only. |
| `--bleu-tokenize` | no | SacreBLEU tokenizer name (default `13a`). |
| `--bleu-lowercase` | no | Lowercase text before BLEU. |
| `--bertscore-lang` | no | BERTScore `lang` preset (default `es`). |
| `--bertscore-batch-size` | no | BERTScore batch size (default `32`). |

## Output JSON (overview)

- **`paper`**: Short provenance (title, Section 3 decoding summary, ref. [40] pointer).
- **`overall`**: Keys **`R_1`**, **`R_2`**, **`R_L`**, **`BLEU`** (0–1), **`BERT`** (mean F1), **`AVG`** (mean of those five, as in Table 4), plus **`diagnostics`** (raw SacreBLEU 0–100 score, precisions, per-example BERT F1).
- **`by_group`**: Optional **`PD`** / **`HC`** blocks with the same structure as **`overall`**.
- **`metrics_config`**: Exact evaluate settings for reproducibility.

## Example

Cmd, PowerShell, or bash (paths repo-relative or absolute):

```bash
python -m main.eval_decode --tokenized-dir data/processed/100k/tokenized --model-path runs/flan-t5-small-baseline/checkpoint-750 --tokenizer-model google/flan-t5-small --output-json runs/flan-t5-small-baseline/test_decode_metrics.json
```

## Notes

- First **BERTScore** run downloads the multilingual BERT weights used for `lang='es'`; allow time and disk.
- Table 4 in the paper is computed on **100** real reports; your `test` row count should match your prepared split (often 96 in this repo’s canonical CSVs—compare before interpreting gaps vs. the paper).
- The PDF cites a **general MT metrics survey** [40] but does not specify a custom BLEU script; this implementation follows **Hugging Face `evaluate`** defaults for comparability. Override **`--bleu-tokenize`** if you need to match another toolkit.
