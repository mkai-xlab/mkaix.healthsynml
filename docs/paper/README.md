# Reference Papers

Source papers kept locally so the literature review stays verifiable offline. The review that cites
them is [`../report/literature_review_cnn_kl_grading.md`](../report/literature_review_cnn_kl_grading.md).

| File | Paper | Relevance |
| --- | --- | --- |
| [`Multimodal Machine Learning-based Knee osteoarthritis progression prediction from plain Radiographs and clinical Data.pdf`](Multimodal%20Machine%20Learning-based%20Knee%20osteoarthritis%20progression%20prediction%20from%20plain%20Radiographs%20and%20clinical%20Data.pdf) | Tiulpin et al. (2019), *Scientific Reports* 9:20038 | Cited in the review. Benchmark for CNN-based KL grading on OAI data; the source of the transfer-learning framing this project follows |
| [`fmed-12-1707588.pdf`](fmed-12-1707588.pdf) | *Potential of AI-based diagnostic grading system for knee osteoarthritis*, Frontiers in Medicine 12:1707588 | Background reading on clinical deployment of automated KL grading. Text extracted to [`fmed-12-1707588.md`](fmed-12-1707588.md) |
| [`diagnostics-16-00539-v2.pdf`](diagnostics-16-00539-v2.pdf) | Almusa et al., *Hybrid Ensemble Model for Knee Osteoarthritis Grading: Integrating CNNs with GLCM Features and XAI*, MDPI Diagnostics | Background reading on CNN ensembles with explainability for KL grading — the same problem shape as this project |

## A note on the extracted markdown

`fmed-12-1707588.md` is an automated text extraction of the PDF next to it, kept for grepping. Its
internal `#page-N` links do not resolve — that is an artefact of the extraction tool, not a broken
document. Read the PDF for anything that matters.

Only the Tiulpin paper is cited in the review; the other two are background reading retained for
provenance.
