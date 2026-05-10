## Proposed Pipeline

```mermaid
%%{init: {'themeVariables': { 'fontSize': '28px' }, 'flowchart': { 'htmlLabels': true, 'wrap': true }}}%%
graph TD
  A["Audio<br/>(4 tasks: /a/, /pa-ta-ka/,<br/>reading, conversation)"] --> B["Preprocess:<br/>16 kHz, scale to -1 to 1"]
  B --> C["Compute acoustic biomarkers:<br/>phonation, prosody, phonemic"]
  C --> D["Select features + map<br/>to mFDA categories"]
  D --> E["Tokenize:<br/>task tags, category headers,<br/>normalized feature tokens"]

  subgraph Synthetic data generation
    C --> S1["Fit per-task/per-category<br/>feature distributions"]
    S1 --> S2["Sample biomarker vectors"]
    S2 --> S3["Create paired simulated<br/>mFDA reports"]
  end

  E --> T["T5/Flan-T5 + LoRA<br/>fine-tuning"]
  S3 --> T

  T --> I["Generated mFDA-style<br/>clinical report"]

  subgraph Evaluation
    I --> V1["BLEU on 100 real reports"]
    I --> V2["Per-category coverage checks"]
    I --> V3["Optional blinded<br/>clinician review"]
  end
```

Figure: End-to-end pipeline from audio to tokenized biomarker prompts and fine-tuned T5/Flan-T5 outputs, with a synthetic data branch for scalable supervision and evaluation on held-out real reports.


