# Configuration for training GPT-2 with RoPE (Rotary Position Embeddings)
# Entraîner avec: python train.py config/train_gpt2_rope.py

# I/O
out_dir = 'out-gpt2-rope'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = False
wandb_project = 'gpt2-rope'
wandb_run_name = 'gpt2-rope'

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# RoPE settings
use_rope = True  # Enable Rotary Position Embeddings
rope_theta = 10000.0  # Base theta (default: 10000.0, can try 5000.0, 20000.0, etc.)

# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# DDP settings
backend = 'nccl'

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
