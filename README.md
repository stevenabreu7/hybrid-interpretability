# hybrid-interpretability

Repository for mechanistic interpretability on hybrid SSM-transformer LLMs.
This repository includes all necessary code to reproduce the results from [Zani et al. (2025)](https://openreview.net/pdf?id=TGWzg86kYv).

<a target="_blank" href="https://colab.research.google.com/drive/1Uq5ByRtrxAWCq7LFSzxsEMekaoQcg1jm?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Setup

**Python 3.10 >= required**

```bash
uv sync
```

Make sure to adjust the paths in the NIAH config as you follow the notebook.

The `Retrieval_with_Sparse_Attention_in_RG.ipynb` notebook shows how to load and inference our modified recurrent gemma model, as well as run the NIAH test. This code is also in `main.py`. 

## Setup (colab)

On colab, uncomment the first cell in `Retrieval_with_Sparse_Attention_in_RG.ipynb`. You should be able to run the notebook with this.

## HF and Kaggle

There is a Kaggle and a HF implementation. During testing, the HF implementation should unusual behavior in the retrieval map, which looked like it did not use the sliding window attention. After developing the Kaggle implementation, it was clear that RG on HF is behaviorally different to Kaggle. We stuck with Kaggle for the paper.
