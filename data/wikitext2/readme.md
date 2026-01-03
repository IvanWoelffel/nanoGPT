## WikiText-2 dataset

After running `prepare.py`:

- train.bin is ~4.63 MB, val.bin ~0.48 MB
- train has ~2.43 M tokens
- val has ~251 K tokens

This came from 36,718 training documents and 3,760 validation documents in total.

The WikiText dataset is a collection of over 2 million tokens extracted from the set of verified Good and Featured articles on Wikipedia.

References:
- WikiText paper: Merity et al. (2016) "Pointer Sentinel Mixture Models"
- HuggingFace: https://huggingface.co/datasets/wikitext
- Original: https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
