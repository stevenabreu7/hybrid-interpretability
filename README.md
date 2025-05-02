# hybrid-interpretability

Repository for mechanistic interpretability on hybrid SSM-transformer LLMs.
This repository includes all necessary code to reproduce the results from [Zani et al. (2025)](https://openreview.net/pdf?id=TGWzg86kYv).

<a target="_blank" href="https://colab.research.google.com/drive/1FfKK23VeDbpoY2IhlhAtR_0QtuBr9Dtf?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Setup

**Python 3.10 >= required**

```bash
uv sync
```
OR
```bash
pip install -r requirements.txt
```

Make sure to adjust the paths in the NIAH config as you follow the notebook.

The `Retrieval_with_Sparse_Attention_in_RG.ipynb` notebook shows how to load and inference our modified recurrent gemma model, as well as run the NIAH test. The benchmark execution code is also in `main.py`. 

## Setup (colab)

On colab, uncomment the first cell in `Retrieval_with_Sparse_Attention_in_RG.ipynb`. You should be able to run the notebook with this.

## Setup (local machine)
`main.py` hosts a script that runs the NIAH benchmark. Make sure to adjust the config file for the benchmark in NIAH/Needle_test/config.yaml to run the right k values. The path to the Kaggle model is automatically set in the script and does not have to be changed.

## HF and Kaggle

There is a Kaggle and a HF implementation. During testing, the HF implementation showed unusual behavior in the retrieval map, which looked like it did not use the sliding window attention. After developing the Kaggle implementation, it was clear that RG on HF is behaviorally different to Kaggle. We stuck with Kaggle for the paper.
