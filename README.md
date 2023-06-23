# RLHF Tutorial
This code is based on this [hugging face tutorial](https://huggingface.co/blog/trl-peft).  
The goal of this tutorial is to fine-tune LM with RLHF in single 24GB GPU.  

## Requirements
* python>=3.9  # >=3.7 might work but haven't been tested
* torch==1.13.1
* `pip install -r req.txt`
* 24GB GPU   # tested in 3090

## File descriptions
You need to execute the files in the following order.
1. sft.ipynb : supervised fine-tune [opt-2.7B](https://huggingface.co/facebook/opt-2.7b) on [imdb](https://huggingface.co/datasets/imdb) data.
2. merge.ipynb : merge fine-tuned low-rank layers into OPT backbone network and re-initialize low-rank layers.
3. rlhf.ipynb : finally does RLHF to maximize reward from pre-trained [sentiment classifier](https://huggingface.co/lvwerra/distilbert-imdb])
