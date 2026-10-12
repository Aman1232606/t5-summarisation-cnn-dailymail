# T5-small News Summarisation: Evaluation against a Lead-3 Baseline

MSc dissertation experiment: how well does a pre-trained T5-small model summarise news articles without any fine-tuning, compared with a simple Lead-3 baseline (the first three sentences of the article)?

## Setup

| Item | Value |
|---|---|
| Model | `t5-small` (Hugging Face Transformers), used without fine-tuning |
| Dataset | CNN/DailyMail 3.0.0 (non-anonymised), test split, 11,490 articles |
| Sample | 200 articles, randomly selected (seed 42) |
| Input / output limits | 512 input tokens, 40-150 output tokens |
| Decoding | Beam search (4 beams), length penalty 2.0, no-repeat 3-grams, repetition penalty 1.1 |
| Metric | ROUGE-1, ROUGE-2, ROUGE-L (F1, with stemming) |
| Baseline | Lead-3: first 3 sentences of the article (NLTK sentence splitter) |
| Hardware | GPU in Google Colab |

## Results (200 articles, mean F1 with bootstrap 95% confidence interval)

| Metric | T5-small | Lead-3 | T5 minus Lead-3 (paired) |
|---|---|---|---|
| ROUGE-1 | 0.385 (0.369-0.402) | 0.403 (0.386-0.420) | -0.018 (-0.033 to -0.004) |
| ROUGE-2 | 0.164 (0.148-0.181) | 0.172 (0.156-0.190) | -0.008 (-0.022 to +0.005) |
| ROUGE-L | 0.267 (0.251-0.284) | 0.250 (0.236-0.267) | +0.017 (+0.003 to +0.031) |

Mean generation time: 1.16 seconds per summary on a GPU.

## Findings

- T5-small scored slightly **below** the Lead-3 baseline on ROUGE-1. The paired confidence interval excludes zero, so the gap is unlikely to be chance, although it is small.
- On ROUGE-2 there is no clear difference (the interval includes zero).
- On ROUGE-L, T5-small scored slightly **above** Lead-3, again by a small margin.
- A simple extractive baseline is therefore hard to beat on this dataset: news articles put the key facts in their opening sentences, which Lead-3 exploits directly.

## Limitations

- Only 200 articles and one model size (`t5-small`); results may differ with larger models or more data.
- T5 checkpoints were pre-trained on a task mixture that included CNN/DailyMail summarisation, so "without fine-tuning" is not a strict zero-shot setting.
- Inputs were truncated to 512 tokens, so later parts of long articles were not seen by the model.
- ROUGE measures word overlap, not factual accuracy or readability. No human evaluation is included in this repository.
- ROUGE-L here is the standard `rougeL` score, not `rougeLsum`, so it should not be compared directly with numbers from papers that report `rougeLsum`.

## How to run

1. Open `summarisation_200.ipynb` in Google Colab.
2. Choose **Runtime > Change runtime type > GPU**.
3. Choose **Runtime > Run all**. It takes roughly 5-10 minutes.
4. The final cell prints a results summary; result tables, figures and run metadata are saved next to the notebook.

Dependencies are listed in `requirements.txt`.
