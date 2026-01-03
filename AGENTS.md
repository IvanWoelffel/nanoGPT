# nanoGPT - Agent Guidelines

## Installation
```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## Commands

### Training
```bash
python train.py config/train_shakespeare_char.py
python train.py --batch_size=32 --compile=False --max_iters=5000
python train.py --device=cuda --batch_size=12 --block_size=1024
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=IP --master_port=1234 train.py
```

### Sampling
```bash
python sample.py --out_dir=out-shakespeare-char
python sample.py --init_from=gpt2-xl --start="What is AI?" --num_samples=5
```

### Data Preparation
```bash
python data/shakespeare_char/prepare.py
python data/shakespeare/prepare.py
python data/openwebtext/prepare.py
python data/bookcorpus/prepare.py
python data/wikitext2/prepare.py
```

### Benchmarking
```bash
python bench.py --batch_size=12 --block_size=1024
```

### Testing
```bash
python train.py --max_iters=100 --eval_iters=10 --log_interval=1
python -c "import numpy as np; m = np.memmap('data/shakespeare_char/train.bin', dtype=np.uint16, mode='r'); print(f'Tokens: {len(m):,}')"
```

## Code Style

**Formatting**: 4 spaces, ~100 chars/line, docstrings `""" """`

**Imports**: Standard lib → Third-party → Local, blank lines between sections

**Naming**: `snake_case` functions/vars, `PascalCase` classes, `lowercase_with_underscores` config, `_private` methods

**Types**: Minimal hints, `@dataclass` for config only

**Logging**: `print(f"{var:,}")` for numbers, `wandb_log = True` optional

**Error handling**: `assert` for invariants with messages

**Config**: Module-level vars, `exec(open('configurator.py').read())`, `--key=value` args

**PyTorch**: `torch.nn.Module`, `torch.nn.functional`, `torch.compile()`, `bfloat16`/`float16`, GradScaler, DDP

**Data**: `numpy.memmap`, `np.uint16`, `tiktoken` GPT-2 BPE, huggingface `datasets`

**Patterns**: model.py (~300 lines), train.py (~300 lines), data/*/prepare.py, @dataclass config, @torch.no_grad()

```python
# Data handling
train_ids = np.array(train_ids, dtype=np.uint16)
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')

# Config dataclass
@dataclass
class GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
```

## File Structure
```
.
├── train.py              # Main training script
├── model.py              # GPT model definition
├── sample.py             # Text generation script
├── bench.py              # Benchmarking script
├── configurator.py        # Config override mechanism
├── config/               # Training configurations
│   ├── train_gpt2.py
│   └── train_shakespeare_char.py
└── data/                 # Dataset directories
    ├── bookcorpus/
    │   └── prepare.py
    ├── openwebtext/
    │   └── prepare.py
    ├── shakespeare/
    │   └── prepare.py
    ├── shakespeare_char/
    │   └── prepare.py
    └── wikitext2/
        ├── prepare.py
        └── readme.md
```

## Principles
Simplicity over abstraction, explicit over implicit, performance (DDP, memmap, multiprocessing), educational value, minimal dependencies

## References
- Karpathy's GPT: https://www.youtube.com/watch?v=kCc8FmEb1nY
- GPT-2 paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- OpenWebText: https://skylion007.github.io/OpenWebTextCorpus/
