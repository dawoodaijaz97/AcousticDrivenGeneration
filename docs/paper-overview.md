
## Abstract:

Speech disorders are rarely analyzed using medical reports. Due to absence of standardize reports.
One of the speech assesgment is the Frenchay Dysarthria Assessment (mFDA) scale to quantify the severity of symptoms across 7 categories
reathing, lips, larynx, palate, monotonicity, tongue, and intelligibilit

In this paper it is proposed to use LLM to generate FDA like reports from audio recording

Used speech recordings from 50 Parkinson’s disease (PD) patients and 50 healthy controls (HC)

The acoustive biomarkers are extracted from speech signal and reports are generated

The results demonstrate that the LLMs can generate reports with a BLEU score of 0.789 for
PD and 0.836 for HC



### Introduction:
Speech is seldom evaluated in PD. PD patients develope dysarthria (Dysarthria is a motor speech disorder caused by weakness, slowness, or poor coordination of the muscles used for speaking, due to neurological injury or disease0)

Frenchay Dysarthria Assessment–2 can also be used to quantify the symtoms 


FDA-2: swallowing, respiration, lips movement,
palate related sounds, laryngeal function (during speech), tongue movement, fluency, and speech intelligibility

mFDA: Listening to recordings. 

breathing, lip movement, larynx function, palate-produced sounds,
monotonicity, tongue movements, and overall intelligibility


### Problem:
however, interpretation can be difficult and requires expert knowledge in signal processing. On the other
hand, clinical assessment can be carried out using perceptual scales such as the mFDA; which is a subjective and demanding
task.

### propose:
combining large language models (LLMs) with acoustic analysis to generate a
clinical report-like text that takes acoustic biomarkers as the primary source of information

## Related work:

Research in automatic medical report generation (aMRG) is not necessarily novel. 


speech has been used as a tool to generate reports, e.g., as spoken instructions instead of text

this is the first work of its kind to extend research on aMRG to the pathological speech domain

### Advancements

CLIP-based models align modality-text pair embeddings through contrastive learning, enabling efficient cross-modal retrieval
and understanding

multimodal LLMs integrate multiple modalities directly into the language model’s embeddings
using modality-specific encoders

Commonly deployed 

The commonly deployed network architecture typically leverages an encoder-decoder
structure, where the encoder is used for domain-specific feature extraction and the decoder for the actual report generation



## Contributions of this paper

extracting features from recordings of PD and HC subjects and tokenizing them as text instructions for the LLM. 

assigned short descriptions (based on the descriptions of the original FDA–2) to each score instead of numerical values to improve
interpretation. 


The mFDA scores were obtained from three phoniatricians who listened and evaluated the audio recordings of
PD patients and HCs across four speech tasks: sustained phonation (vowel /a/), repetition of /pa-ta-ka/ (DDK), text reading,
and a spontaneous conversation with the interviewer/evaluator.

We computed several acoustic biomarkers and selected the
most relevant features per mFDA category (breathing, lips, palate, larynx, monotonicity, tongue, and intelligibility) using
an statistical approach. 


Given the limited amount of subjects in our work, we used the simulated reports to help the model learn
to interpret the input prompts and also to create the mFDA-based reports following a structured format. To “simulate" the
acoustic biomarkers, we use the distribution of the real data


### main contribution
1.A speech-based approach that takes acoustic biomarkers as prompts to an LLMs that generate mFDA-based text reports,
improving interpretability and user controls over the input used by the language model to generate text.

2.A data augmentation strategy to simulate acoustic biomarkers and clinical reports. The primary function of such reports
is for the model to learn to interpret acoustic-based input prompts and to generate structured speech reports


### main goal

Our main goal is to show how LLMs can be used to support clinical experts to assess patients with speech technology

### research area
The research question is whether LLMs can effectively learn to associate acoustic biomarkers with clinical speech
assessment reports

### motivation

The motivation behind this work is the almost non-existent approaches in the literature that consider the audio signals as
the primary source of information to generate a clinical-like report, instead of using speech as a tool to input spoken prompts to
a LLM or to improve the recognition rate on an automatic speech recognition system.

## materials and methods



## Data

We considered speech recordings of 50 PD patients (25 male / 25 female) and 50 HC (25 male / 25 female) Colombian Spanish
native speakers from the PC-GITA dataset


The average age of the PDs is 61.9± 9.61 and 60.9±9.46 for HCs.

All audios are resampled at 16kHz. 

We considered four speech tasks: (1) sustained phonation
of vowel /a/, (2) repetition of /pa-ta-ka/, (3) text reading, and (4) spontaneous conversation with the interviewer/evaluator..

The mFDA has seven categories, which an expert
phoniatrician rates with numbers between “0” (normal) and “4” (severe impairment).

We use the median value of
the scores given by the phoniatricians

In this work, we associated the scores to a categorical rating as follows: 0/1: Normal,
2: Mild, 3: Moderate, and 4: Severe


## Acoustic Biomarkers

We computed 248 task-specific (Section 2) acoustic biomarkers
no denosing was performed; however, we scaled the amplitude
of the audios between -1 and 1 and centered them around zero. 

### phonation feature:

reflects voice quality, stability,

Pitch (F₀) Standard Deviation
Sound Pressure level
Jitter
Shimmer

### Prosody features
Prosody encompasses the rhythm, intonation, stress, and melody of speech

Pitch & SPL standard deviation
Pause Duration, Rate, Count
Voiced vs. Voiceless Segment Ratios


### Phonemic features

Phonemic features describe the precision and coordination of articulators


Pronunciation Precision (Posterior Probabilities, Log-Likelihood Ratio)
Duration & Rate of Phoneme Classes
Pairwise Variability Index (PVI)
Place & Manner of Articulation Groups


Feature Group	Focus	mFDA Categories Mapped	Speech Task Used
Phonation	Voice source stability & loudness	Larynx	Sustained vowel /a/
Prosody	Melody, rhythm, timing	Monotonicity, Breathing	Reading & Conversation
Phonemic	Articulatory precision	Lips, Tongue, Palate, Intelligibility	/pa-ta-ka/ repetition, Reading

## Language models

The mFDA-based clinical reports are generated by fine-tuning several pre-trained LLMs from Hugging Face2: T5 (TextTo-Text Transfer Transformer)33, Flan-T5 (Fine-tuned Language Net-T5)34, GPT-2 (Generative Pre-trained Transformer)35,
LLaMA-7B36, Mistral-7B37, and DeepSeek-R1-Qwen2.5-7B38.

## Training
Key challenge was fine tuning without overfitting and having sufficient data to teach the model

we designed an augmentation strategy based on the acoustic biomarkers of the real data to
simulate input prompts and their corresponding mFDA reports

Extracted acoustic biomarkers correlated with mFDA categories.

Simulated large-scale paired data (biomarker → clinical report) using statistical distributions of the real features.

Fine-tuned several LLMs (T5, Flan-T5, GPT-2, LLaMA-7B, Mistral-7B, DeepSeek-Qwen 7B) on these simulated pairs.

Evaluated them on the 100 real mFDA reports from the actual dataset.



## Experiment and results:
This produced a synthetic dataset of 110 000 (prompt, report) pairs.


Training set: 100 000 pairs

Validation set: 10 000 pairs

We randomly generated 110K instructions (input prompts) with associated mFDA-based clinical reports (target output) with the
following conditions: 25% of the reports contain random ratings on all categories, 25% contain only Normal-Mild ratings, 25%
contain only Mild-Moderate rating, and 25% contain only Moderate-Severe ratings. 

Real data (50 PD + 50 HC) was reserved only for testing, to prevent overfitting.

The models were trained on an NVIDIA A100 with 40 GB over three epochs. The learning rate was set to
3e-4, the weight decay to 10e-2, and the batch size was 8. 


## Conclusion:

This study introduced a novel framework for generating mFDA-based clinical speech reports from acoustic biomarkers using large language models (LLMs). The approach bridges quantitative acoustic analysis and qualitative clinical interpretation, improving both interpretability and control of automated speech assessments.