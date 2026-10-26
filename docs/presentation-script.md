# Presentation Script — Acoustic-Driven Generation of mFDA Clinical Speech Reports with Flan-T5

Source deck: `docs/AcousticDrivenGeneration_Presentation.pptx` (19 slides).
Target length: **15–20 minutes**, leaving time for Q&A after slide 19.
Pacing guide below sums to ~19 minutes at a natural pace — if you're
running long, slides 10 and 14 are the easiest to compress.

Speak in full sentences during rehearsal, but on the day, use the bullets
on the slide as your outline — don't read this script verbatim. Bold text
= the number/phrase to land hardest on.

---

## Slide 1 — Title (~30s)

> Good [morning/afternoon], everyone. I'm Dawood Aijaz, and today we'll be
> presenting our project work: **acoustic-driven generation of mFDA clinical
> speech reports using Flan-T5** — teaching a compact language model to
> turn seven numeric acoustic biomarkers into a readable clinical
> assessment of Parkinsonian speech.

*(Reminder: the slide still says "Supervisor: (add name)" — fill that in
before presenting.)*

---

## Slide 2 — Agenda (~25s)

> Here's the roadmap. We'll start with the motivation and the reference
> paper we're building on, then walk through our data and pipeline, the
> model we trained. Then the most important part of this talk — our
> evaluation methodology, because it turns out *how* you measure this
> problem matters as much as the model itself. We'll close with results,
> limitations, and where this goes next.

---

## Slide 3 — Section divider: "From Biomarkers to Clinical Reports" (~10s)

> So — from biomarkers to clinical reports. What does that actually mean?

*(Brief pause, let the section title land, move straight into slide 4.)*

---

## Slide 4 — Why automate mFDA speech assessment? (~70s)

> Parkinson's disease affects speech early and measurably — breathing
> support, articulation, monotone pitch — but in clinical practice this is
> **rarely assessed formally**. The modified Frenchay Dysarthria
> Assessment, or mFDA, is the standard scale: seven categories, each rated
> by a trained clinician. That works, but it's **subjective, slow, and
> inconsistent** between raters.
>
> At the same time, we *can* extract objective acoustic biomarkers
> directly from a speech recording — but reading those numbers requires
> signal-processing expertise most clinicians don't have.
>
> So the goal of this project is simple to state: build a system that
> **bridges that gap** — take the biomarkers, and have a language model
> generate the readable clinical report a specialist would have written.

---

## Slide 5 — Reference study: Arias-Vergara et al., 2025 (~75s)

> This isn't a from-scratch idea — we're reproducing and extending a 2025
> paper that frames this exact problem as **text-to-text generation**:
> seven biomarker scores in, a fine-tuned LLM in the middle, an mFDA
> report out.
>
> The original authors compared a whole spread of models — T5, Flan-T5,
> GPT-2, and three **7-billion-parameter** models: LLaMA-7B, Mistral-7B,
> and DeepSeek-7B. Their headline result was a BLEU of **0.79 on
> Parkinson's patients and 0.84 on healthy controls**, from those large
> models.
>
> Our focus is deliberately narrower: **Flan-T5**. It's compact, it fits
> on a single GPU, and — critically for a clinical tool — it's realistic
> to deploy on-premises in a hospital, where you can't always ship patient
> speech data to a cloud API.

---

## Slide 6 — Data: PC-GITA + synthetic augmentation (~75s)

> Where does the training data come from? The underlying corpus is
> **PC-GITA** — 50 Parkinson's patients and 50 healthy controls, each
> recorded across four speech tasks, giving us 248 acoustic biomarkers
> per speaker, which map down to the seven mFDA categories.
>
> Here's the catch: we only have **100 real clinical reports** — nowhere
> near enough to fine-tune a language model on. So, following the paper's
> approach, we **simulate** the rest: roughly **110,000 synthetic
> biomarker-to-report pairs** generated programmatically from the
> biomarker distributions.
>
> The real reports are precious, so we don't train on them at all — we
> hold out **96 of them purely for testing**. That's an important design
> choice: the model learns from synthetic data, but it's judged only
> against real clinical writing.

*(Point to the data diagram on screen while explaining the split.)*

---

## Slide 7 — Pipeline: one ETL core parses every biomarker, once (~55s)

> To keep all of this reproducible, everything runs through **one
> shared pipeline**, three stages: ETL turns raw CSVs into a canonical
> dataset, a preparation stage tokenizes it for the model, and then
> training and evaluation consume that.
>
> The key design decision is the ETL core — there's exactly **one place**
> in the codebase that parses the raw biomarker strings and builds the
> input/output text pairs. Every downstream script — training data,
> synthetic augmentation, evaluation — calls that same parser. That
> matters for correctness: we never have two slightly-different regexes
> silently disagreeing about what a biomarker string means.

---

## Slide 8 — What is Flan-T5? (~75s)

> Quick primer before we get into our specific model. **T5** — Text-to-Text
> Transfer Transformer — is Google's encoder-decoder transformer that
> frames every NLP task as text in, text out. That's exactly the shape of
> our problem: **biomarker scores as text in, a clinical report as text
> out**.
>
> **Flan-T5** takes that same architecture and further fine-tunes it on
> over a thousand tasks phrased as natural-language instructions — the
> "Flan" stands for **F**ine-tuned **LA**nguage **N**et. That instruction
> tuning gives it a head start at following a prompt, before we ever
> fine-tune it further ourselves.
>
> It comes in five sizes: **Small at 80 million** parameters, **Base at
> 248 million**, **Large at 780 million**, **XL at 3 billion**, and **XXL
> at 11 billion**. We use **Base** — small enough to fine-tune and deploy
> on a single GPU, large enough to learn the task well.

---

## Slide 9 — Best model: B17 = Flan-T5-base + LoRA (~75s)

> So what did we actually train? Our best model — internally labeled
> **B17** — starts from **`google/flan-t5-base`, 248 million parameters**.
> The recipe is two stages: **five epochs of full fine-tuning**, then
> **three more epochs of rank-32 LoRA** on top, with the adapter weights
> merged back into the base model afterward.
>
> The input prompt is intentionally compact — "Generate a report for:"
> followed by the seven biomarker scores — not the raw instruction text.
> We tried the raw-instruction version too; it did worse.
>
> One thing we want to flag: learning rate **5e-4** and weight decay
> **0.01** weren't picked by validation loss. They were picked by
> **actual decode quality** — generating text and scoring it — because in
> this project, loss and generation quality repeatedly disagreed. We'll
> come back to why that matters.

---

## Slide 10 — Experiment setup: compute & scripts (~55s)

> Briefly, on infrastructure: everything ran on the **NHR@FAU TinyGPU
> cluster** — mainly A100 40GB GPUs, with V100 or RTX 3080 as fallback,
> scheduled through Slurm with up to a 24-hour wall-time cap per job.
>
> To keep 20-plus experiments organized, every one has a **matched pair
> of Slurm scripts** — a train script and an eval script — named by
> experiment ID. That's **39 scripts total**, all calling the same two
> fixed entry points, `main.train` and `main.eval_decode`. That
> consistency is what let us compare experiments fairly instead of each
> one being a one-off.

---

## Slide 11 — Why two evaluation scripts? (~90s)

> Now — the methodology slide, and honestly the most important one in
> this talk. We ended up running **two separate evaluation scripts**, and
> here's why.
>
> For day-to-day model selection, we use what we'll call the **honest
> ruler**: word-level SacreBLEU-4, BERTScore with the language set
> correctly to English, and deterministic beam search. It's the
> standardized, widely-comparable way to score generated text.
>
> But when we want to compare directly against the paper's numbers, we
> use a second script that **reproduces their exact evaluation code**:
> **character-level BLEU-2**, not word-level BLEU-4, plus their specific
> sampling-based decoding.
>
> Why does this matter enough to build two scripts? Because **"BLEU" is
> not one number**. Word-level and character-level BLEU can differ by
> **two times** on the *exact same generated text*. If you don't know
> which one a paper used, you cannot honestly compare to it. That
> realization is what the rest of the results section is built on.

---

## Slide 12 — B17 results: two rulers, one model (~85s)

> So here's B17, scored both ways. On our **honest, word-level ruler**,
> the overall average is **0.559**. Read in isolation, that looks well
> below the paper's headline numbers, and for a while, that's exactly how
> we read it.
>
> But score the **same model, same outputs** with the **paper's own
> character-level ruler**, and the average is **0.649**. The paper's best
> 7-billion-parameter systems land at roughly **0.68**.
>
> That's the headline of this project: under the paper's own metric, our
> **248-million-parameter** model comes within about **0.03** of models
> **roughly 28 times larger**. Same weights, same generated text — the
> only thing that changed between 0.559 and 0.649 is the ruler we
> measured it with.

---

## Slide 13 — The "gap" is mostly the metric, not the model (~75s)

> Let's unpack that, because it's a striking enough claim that it needs
> justification, not just a slide.
>
> Two things were happening. First, word BLEU-4 versus character BLEU-2
> on identical output text differ by **nearly two times** — for B17
> specifically, that's **0.36 versus 0.76**. Second, our BERTScore was
> defaulting to a **Spanish-language** backbone model, because the
> underlying speech is Colombian Spanish — but the *reports themselves*
> are written in **English**. Scoring English text with a Spanish model
> silently under-scores it. Fixing the language argument alone moved
> BERTScore from 0.823 to 0.911.
>
> Neither of those fixes changed a single generated word. They're both
> purely measurement corrections — which is exactly why we're confident
> calling this a metric artifact rather than a model improvement.

*(Reference the chart — it shows the same outputs under both rulers side
by side.)*

---

## Slide 14 — What we tried — and closed (~65s)

> Before settling on B17, we ran over **20 controlled experiments** on
> Flan-T5-base, changing one variable at a time. The single biggest lever
> was **learning rate** — that's what separated a mediocre model from a
> good one. LoRA rank, epoch count, and weight decay each gave small,
> incremental gains that we've now closed out.
>
> A few things we hoped would help, didn't: forcing structured,
> slot-by-slot targets instead of free text, feeding the model the raw
> instruction text instead of a compact score string, and simply scaling
> up to Flan-T5-*large*. All of them **regressed**. That matters for
> anyone extending this work — those are dead ends we've already ruled
> out, not untested ideas.

---

## Slide 15 — Persistent gap: PD reports are harder (~70s)

> One gap did *not* disappear when we fixed the metric: **Parkinson's
> patient reports are consistently harder to generate than healthy
> control reports**, on every metric we tried. That matches the original
> paper's own finding, so it's not an artifact of our pipeline — it looks
> like a real property of the task. PD reports are more varied and
> clinically complex, which is intuitively why.
>
> The other honest finding here: when we tried to *force* the model to
> produce all seven mFDA categories in a strict structured format, only
> about **14%** of generations actually contained all seven slots — versus
> **100%** in the reference reports. And forcing that structure didn't
> just fail to help — it actively **hurt fluency**. That's a real,
> unsolved limitation, not something the metric fix explains away.

---

## Slide 16 — Limitations & clinical considerations (~70s)

> A few limitations worth being upfront about. First — and this applies
> to the whole field, not just us — **these metrics measure text
> overlap, not clinical correctness**. A report can score well on BLEU
> and BERTScore while still being clinically wrong; we have not validated
> clinical accuracy with a human rater.
>
> Second, our synthetic training data is **highly templated** — a
> duplicate rate of **0.95** — which likely makes the generation task
> easier than it would be on more naturalistic reports. Third, this is a
> **single corpus, single language**: Colombian-Spanish speakers only, so
> generalization to other populations is untested. And our real test set
> is only **96 reports**, so we'd treat the PD-versus-HC gap as a real
> signal, but hold the exact numbers loosely.
>
> Given all of that, we want to be explicit about positioning: this is
> designed as a **decision-support drafting aid** for a clinician to
> review and edit — **never** an autonomous diagnostic tool.

---

## Slide 17 — Conclusion & next steps (~70s)

> To conclude: **B17 is final for the Flan-T5 track.** We've exhausted
> the tuning levers available to a 248-million-parameter model, and under
> the paper's own metric, it's genuinely competitive with systems nearly
> thirty times its size.
>
> Three directions from here. One: **constrained or template-guided
> decoding**, to close that 14%-structure-coverage gap without hurting
> fluency. Two: **richer, less-templated synthetic training data**, since
> our current data's high duplication rate is a likely ceiling. Three,
> and the most ambitious: **scale to a 7-billion-parameter decoder** —
> LLaMA or Mistral with LoRA — to actually try to beat the paper's 0.79 /
> 0.84 headline, rather than just approach it.

---

## Slide 18 — Bottom line (~25s)

> If you take away one sentence from this talk, let it be this one:
> **under the paper's own metric, a 248-million-parameter model lands
> beside systems roughly 28 times larger — the gap was the ruler, not the
> model.**

*(Pause here. Let it sit before moving to the closing slide.)*

---

## Slide 19 — Thank you (~15s)

> Thank you for your attention — we're happy to take any questions.

---

## Anticipated questions (prep, not part of the script)

- **"Why should I trust the character-BLEU comparison over word-BLEU?"**
  → Neither is more "correct" in the abstract; the point is you must use
  the *same* ruler as whatever you're comparing to. We report both,
  transparently, and label which is which (slide 11–12).
- **"Could the synthetic data's high duplication rate be inflating your
  numbers too, the same way the metric bug did?"**
  → Possibly for the *training* signal, but the test set is 96 **real**
  clinical reports, never seen in training — so the reported decode
  metrics are on genuine held-out clinical text, not synthetic data.
- **"Why not just try the 7B model now?"**
  → Compute/time budget for this project; flagged explicitly as the next
  track, not a tweak to this one (slide 17).
- **"What does 'structure coverage 14%' mean in practice?"**
  → Of generated reports, only 14% mention all seven mFDA categories by
  name in the expected `Category (Severity): description` shape; the
  rest omit or merge some categories, even though content may still be
  clinically reasonable prose.
