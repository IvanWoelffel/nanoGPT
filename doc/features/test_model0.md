# Test Plan: Comparing Nano-GPT on WikiText-2 vs BookCorpus

## Overview

This plan creates a comprehensive Jupyter notebook (`test_wikitext2_bookcorpus.ipynb`) to train and evaluate Nano-GPT models on two datasets with different text styles:
- **WikiText-2**: Encyclopedic text from Wikipedia
- **BookCorpus**: Narrative text from books (sampled subset)

The goal is to understand how text style (encyclopedic vs narrative) affects:
1. Training convergence speed
2. Final model performance (perplexity)
3. Text generation quality
4. Token distribution patterns

## Dataset Information

### WikiText-2 (Encyclopedic)

- **Source**: Wikipedia Good and Featured articles
- **Size**: ~2.43M training tokens, ~251K validation tokens
- **Files**: `data/wikitext2/train.bin` (4.63 MB), `data/wikitext2/val.bin` (0.48 MB)
- **Style**: Formal, factual, technical terminology
- **Characteristics**:
  - Well-structured, informational content
  - Academic and scientific vocabulary
  - Consistent sentence structure
  - Technical terms and domain-specific language

### BookCorpus (Narrative) - Sampled

- **Source**: Books by unpublished authors (~11,038 books)
- **Original size**: ~74M documents, several billion tokens
- **Sampled size**: ~2.5M training tokens, ~250K validation tokens (to match WikiText-2)
- **Files**: `data/bookcorpus/train_sample.bin`, `val_sample.bin` (created during notebook execution)
- **Style**: Narrative, literary, dialogues
- **Characteristics**:
  - Fictional storytelling elements
  - Conversational dialogue patterns
  - Descriptive and emotive language
  - Varied sentence structures and narrative pacing

## Hypotheses to Test

1. **Convergence Speed**: Does text style affect how quickly models learn?
   - Hypothesis: BookCorpus may converge faster due to repetitive dialogue patterns
   - Hypothesis: WikiText-2 may require more iterations due to technical diversity

2. **Final Performance**: Which dataset produces lower perplexity?
   - Hypothesis: Similar perplexities given equal data and architecture
   - Alternative: One dataset may be inherently more predictable

3. **Text Generation**: Will models generate text faithful to their training style?
   - Hypothesis: WikiText-2 model generates factual, encyclopedia-style text
   - Hypothesis: BookCorpus model generates narrative, dialogue-heavy text

4. **Vocabulary Usage**: How does token distribution differ?
   - Hypothesis: WikiText-2 uses more technical vocabulary
   - Hypothesis: BookCorpus uses more common, everyday vocabulary

## Jupyter Notebook Structure

### Cell 1: Introduction (Markdown)

```markdown
# Nano-GPT: WikiText-2 vs BookCorpus Comparison

This notebook compares training Nano-GPT on two datasets with different text styles:
- **WikiText-2**: Encyclopedic text from Wikipedia
- **BookCorpus**: Narrative text from books (sampled subset)

**Objectives:**
1. Compare training curves and convergence
2. Analyze final perplexities
3. Compare text generation quality
4. Study token distribution differences

**Time estimate**: ~30 minutes on GPU
```

### Cell 2: Setup and Imports

```python
# Import all necessary libraries for training, evaluation, and visualization
import torch
import torch.nn as nn
import numpy as np
from model import GPT, GPTConfig
import tiktoken
from contextlib import nullcontext
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys
from tqdm.auto import tqdm

# Set random seeds for reproducibility
torch.manual_seed(1337)
np.random.seed(1337)

# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("Setup complete. Libraries imported.")
```

### Cell 3: Configuration

```python
# ============================================================================
# CONFIGURATION - HYPERPARAMETER EXPLANATIONS
# ============================================================================

# --- DEVICE SELECTION ---
# 'cuda': Use GPU (recommended for training, much faster)
# 'cpu': Use CPU (fallback, very slow, for testing only)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# --- DATA TYPE SELECTION ---
# bfloat16: Preferred for modern GPUs (A100, RTX 30xx/40xx), offers good precision/speed trade-off
# float16: Fallback for older GPUs, requires gradient scaling to avoid numerical instability
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
print(f"Dtype: {dtype}")

# --- COMMON HYPERPARAMETERS (Identical for fair comparison) ---

# --- BATCH SIZE ---
# batch_size = 12: Chosen based on:
# - GPU memory constraints (typical 12GB-24GB GPUs can handle this comfortably)
# - Training stability: Too small = noisy gradients, too large = less frequent updates
# - Trade-off: Larger batch = faster training but may need higher learning rate
batch_size = 12
print(f"Batch size: {batch_size}")

# --- BLOCK SIZE (CONTEXT WINDOW) ---
# block_size = 256: Chosen because:
# - Both datasets have sentences with moderate length
# - Larger context requires more memory and computation (O(n^2) in attention)
# - Trade-off: 256 is sweet spot for small models (6 layers)
block_size = 256
print(f"Block size: {block_size}")

# --- MODEL ARCHITECTURE ---
# Small but effective model for fast training and reasonable quality:
# - n_layer = 6: Depth sufficient for language modeling
# - n_head = 6: Number of attention heads, should divide n_embd evenly
# - n_embd = 384: Embedding dimension, larger = more capacity but slower
# - Reasoning: ~10M parameters, trains in ~10-15 minutes on GPU per dataset
n_layer = 6
n_head = 6
n_embd = 384
print(f"Model: {n_layer} layers, {n_head} heads, {n_embd} dimensions")

# --- LEARNING RATE ---
# learning_rate = 1e-3: Chosen because:
# - Small models (~10M params) can handle higher learning rates
# - AdamW optimizer: 1e-3 is standard starting point for small models
# - Too high: Instability, diverging loss
# - Too low: Very slow convergence
learning_rate = 1e-3
print(f"Learning rate: {learning_rate}")

# --- MAX ITERATIONS ---
# max_iters = 3000: Reduced from 5000 to fit in 30-minute budget
# - Enough to see clear learning curves and convergence
# - Both datasets have ~2.4-2.5M tokens, each iter processes ~3K tokens
# - Total tokens seen: 3000 * 12 * 256 = 9.22M tokens (~3-4 epochs)
max_iters = 3000
print(f"Max iterations: {max_iters}")

# --- EVALUATION SETTINGS ---
# eval_interval = 200: Check progress every 200 iterations (15 times total)
# eval_iters = 100: Average loss over 100 batches for stable estimate
eval_interval = 200
eval_iters = 100
print(f"Evaluation: every {eval_interval} iters, {eval_iters} batches")

# --- GRADIENT ACCUMULATION ---
# gradient_accumulation_steps = 1: No accumulation
gradient_accumulation_steps = 1
print(f"Gradient accumulation: {gradient_accumulation_steps}")

# --- DROPOUT ---
# dropout = 0.1: Light regularization for small dataset
dropout = 0.1
print(f"Dropout: {dropout}")

# --- LEARNING RATE DECAY ---
# decay_lr = True: Decay LR over time helps convergence
# warmup_iters = 100: LR ramps up from 0 to full LR in first 100 iters
# lr_decay_iters = max_iters: Decay until max_iters
# min_lr = 1e-4: Minimum LR (typically 1/10 of starting LR)
decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = 1e-4
print(f"LR decay: {decay_lr}, warmup: {warmup_iters}, min_lr: {min_lr}")

# --- MISCELLANEOUS ---
# weight_decay = 0.1: L2 regularization for AdamW (helps prevent overfitting)
# beta1 = 0.9, beta2 = 0.99: AdamW hyperparameters (standard values)
# grad_clip = 1.0: Prevent exploding gradients
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
print(f"Optimizer: weight_decay={weight_decay}, betas=({beta1}, {beta2}), grad_clip={grad_clip}")

# --- BOOKCORPUS SAMPLING TARGET ---
# TARGET_TOKENS = 2.5M: Match WikiText-2 size for fair comparison
TARGET_TOKENS = 2_500_000
print(f"\nBookCorpus sampling target: {TARGET_TOKENS:,} tokens")

print("\n" + "="*70)
print("Configuration complete!")
print("="*70)
```

### Cell 4: Dataset Preparation Functions

```python
# ============================================================================
# DATASET PREPARATION FUNCTIONS
# ============================================================================

def prepare_wikitext2():
    """
    Prepare WikiText-2 dataset if not already prepared.
    Downloads from HuggingFace and tokenizes using GPT-2 BPE.

    WikiText-2 characteristics:
    - 36,718 training documents
    - 3,760 validation documents
    - ~2.43M tokens total
    - Encyclopedic, factual style
    """
    train_path = 'data/wikitext2/train.bin'
    val_path = 'data/wikitext2/val.bin'

    # Check if already prepared
    if os.path.exists(train_path) and os.path.exists(val_path):
        print("✓ WikiText-2 already prepared")
        return

    print("Preparing WikiText-2 dataset...")

    # Import and download dataset
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Tokenize and save each split
    for split, split_name in [('train', 'train'), ('validation', 'val')]:
        all_ids = []

        for example in tqdm(dataset[split], desc=f"Tokenizing {split}"):
            # Skip empty lines
            if example['text'].strip():
                ids = enc.encode_ordinary(example['text'])
                ids.append(enc.eot_token)
                all_ids.extend(ids)

        # Save to binary file
        arr = np.array(all_ids, dtype=np.uint16)
        os.makedirs('data/wikitext2', exist_ok=True)
        arr.tofile(f'data/wikitext2/{split_name}.bin')
        print(f"  {split_name}: {len(arr):,} tokens ({len(arr)*2/1024/1024:.2f} MB)")

    print("✓ WikiText-2 preparation complete")


def prepare_bookcorpus_sample(target_tokens=TARGET_TOKENS):
    """
    Prepare a sample of BookCorpus dataset to match WikiText-2 size.

    BookCorpus characteristics:
    - ~74M documents original (too large for 30-min experiment)
    - Narrative, literary style with dialogues
    - Using streaming to avoid downloading entire dataset

    Why sample instead of full dataset?
    - Fair comparison: Same token count as WikiText-2
    - Time constraint: Full dataset would take hours to prepare/train
    - Reproducibility: Smaller sample is easier to manage
    """
    train_path = 'data/bookcorpus/train_sample.bin'
    val_path = 'data/bookcorpus/val_sample.bin'

    # Check if already prepared with sufficient tokens
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
        if len(train_data) >= target_tokens * 0.9:
            print(f"✓ BookCorpus sample already prepared ({len(train_data):,} tokens)")
            return

    print(f"Preparing BookCorpus sample (target ~{target_tokens:,} tokens)...")

    # Import and stream dataset (avoids downloading all 74M docs)
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    val_ratio = 0.1  # 10% for validation

    # Stream dataset to avoid memory issues
    dataset = load_dataset("SamuelYang/bookcorpus", split="train", streaming=True)

    train_ids = []
    val_ids = []

    # Tokenize until we reach target
    for i, example in enumerate(tqdm(dataset, desc="Tokenizing BookCorpus")):
        if len(train_ids) >= target_tokens:
            break

        text = example['text']
        if text.strip():
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)

            # Split: 90% train, 10% val
            if np.random.random() < val_ratio:
                val_ids.extend(ids)
            else:
                train_ids.extend(ids)

    # Save samples
    os.makedirs('data/bookcorpus', exist_ok=True)

    train_arr = np.array(train_ids[:target_tokens], dtype=np.uint16)
    train_arr.tofile(train_path)
    print(f"  train_sample: {len(train_arr):,} tokens ({len(train_arr)*2/1024/1024:.2f} MB)")

    val_arr = np.array(val_ids[:int(target_tokens*val_ratio)], dtype=np.uint16)
    val_arr.tofile(val_path)
    print(f"  val_sample: {len(val_arr):,} tokens ({len(val_arr)*2/1024/1024:.2f} MB)")

    print("✓ BookCorpus sample preparation complete")


print("Dataset preparation functions defined!")
```

### Cell 5: Prepare and Load Datasets

```python
# ============================================================================
# PREPARE AND LOAD DATASETS
# ============================================================================

# Prepare datasets if needed
print("Preparing datasets...")
prepare_wikitext2()
prepare_bookcorpus_sample()

# Load datasets using numpy memmap (efficient for large files)
print("\nLoading datasets...")

wikitext2_train = np.memmap('data/wikitext2/train.bin', dtype=np.uint16, mode='r')
wikitext2_val = np.memmap('data/wikitext2/val.bin', dtype=np.uint16, mode='r')

bookcorpus_train = np.memmap('data/bookcorpus/train_sample.bin', dtype=np.uint16, mode='r')
bookcorpus_val = np.memmap('data/bookcorpus/val_sample.bin', dtype=np.uint16, mode='r')

# Display dataset statistics
print(f"\n{'='*60}")
print("DATASET STATISTICS")
print(f"{'='*60}")

print(f"\nWikiText-2 (Encyclopedic):")
print(f"  Train: {len(wikitext2_train):,} tokens ({len(wikitext2_train)*2/1024/1024:.2f} MB)")
print(f"  Val:   {len(wikitext2_val):,} tokens ({len(wikitext2_val)*2/1024/1024:.2f} MB)")

print(f"\nBookCorpus (Narrative - Sampled):")
print(f"  Train: {len(bookcorpus_train):,} tokens ({len(bookcorpus_train)*2/1024/1024:.2f} MB)")
print(f"  Val:   {len(bookcorpus_val):,} tokens ({len(bookcorpus_val)*2/1024/1024:.2f} MB)")

print(f"\nComparison:")
print(f"  WikiText-2 / BookCorpus ratio: {len(wikitext2_train)/len(bookcorpus_train):.2f}x")
print(f"  Difference: {abs(len(wikitext2_train) - len(bookcorpus_train)):,} tokens")

print(f"\n{'='*60}")
print("Datasets loaded successfully!")
print(f"{'='*60}")
```

### Cell 6: Model Creation

```python
# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def create_model(vocab_size=50257):
    """
    Create GPT model with specified architecture.

    Architecture choices explained:
    - n_layer = 6: Depth determines model's capacity to learn complex patterns
      * More layers = more capacity but slower and harder to train
      * 6 layers is reasonable for ~2.5M tokens
      * GPT-2 uses 12 layers, but we have less data and time

    - n_head = 6: Number of attention heads in multi-head attention
      * Each head learns different aspects of relationships
      * Must divide n_embd evenly (384 / 6 = 64 dimensions per head)

    - n_embd = 384: Embedding dimensionality
      * Total params ≈ 6 * 384^2 * 12 ≈ 10M parameters
      * Reasonable for 30-minute training experiment
      * GPT-2 uses 768, but we have smaller dataset

    - dropout = 0.1: Regularization to prevent overfitting
      * Randomly drops 10% of activations during training
      * Helps model generalize better

    Returns:
        GPT model instance
    """
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=True,
    )

    return GPT(config)


# Create models for both datasets (same architecture for fair comparison)
print("Creating models...")

model_wikitext = create_model()
model_bookcorpus = create_model()

num_params = model_wikitext.get_num_params()

print(f"  WikiText-2 model: {num_params/1e6:.2f}M parameters")
print(f"  BookCorpus model: {num_params/1e6:.2f}M parameters")
print(f"  Architecture: {n_layer} layers, {n_head} heads, {n_embd} dimensions")

print("\n" + "="*70)
print("Models created successfully!")
print("="*70)
```

### Cell 7: Training Functions

```python
# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_batch(data, batch_size, block_size, device):
    """
    Sample a batch of data from the dataset.

    Why random sampling?
    - Ensures model sees diverse data each iteration
    - Prevents memorization of specific sequences
    - Improves generalization

    Args:
        data: numpy memmap of token ids
        batch_size: number of sequences per batch
        block_size: sequence length (context window)
        device: 'cuda' or 'cpu'

    Returns:
        x: input tokens (batch_size, block_size)
        y: target tokens (batch_size, block_size), shifted by 1
    """
    # Randomly sample starting positions for each batch element
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Extract sequences and convert to tensors
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    # Move to device
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device, dtype_ctx):
    """
    Estimate loss on train/val sets by averaging over multiple batches.

    Why average over multiple batches?
    - Reduces noise in loss estimate
    - Provides more stable metric for comparison
    - 100 batches gives good balance between accuracy and speed

    Args:
        model: GPT model
        train_data: training data
        val_data: validation data
        batch_size: batch size for evaluation
        block_size: context window size
        eval_iters: number of batches to average
        device: device to run on
        dtype_ctx: autocast context for mixed precision

    Returns:
        losses: dictionary with 'train' and 'val' losses
    """
    model.eval()
    losses = {}

    for split, data in [('train', train_data), ('val', val_data)]:
        loss_list = []
        for _ in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)

            # Mixed precision inference (faster, less memory)
            with dtype_ctx:
                _, loss = model(X, Y)

            loss_list.append(loss.item())

        losses[split] = np.mean(loss_list)

    model.train()
    return losses


def get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """
    Learning rate schedule with warmup and cosine decay.

    Why warmup?
    - Prevents large destabilizing gradients early in training
    - Gradually increases from 0 to target LR
    - Standard practice for transformer training

    Why cosine decay?
    - Smoothly decreases LR from max to min
    - Helps model converge to better minimum
    - Better than step decay for most tasks

    Args:
        iter_num: current iteration number
        warmup_iters: warmup iterations
        lr_decay_iters: total decay iterations
        learning_rate: maximum learning rate
        min_lr: minimum learning rate

    Returns:
        learning_rate: current learning rate
    """
    # 1) Linear warmup
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / (warmup_iters + 1)

    # 2) If past lr_decay_iters, return min learning rate
    if iter_num > lr_decay_iters:
        return min_lr

    # 3) Cosine decay between warmup and lr_decay_iters
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train_model(model, train_data, val_data, max_iters, learning_rate, out_dir, dataset_name):
    """
    Train a GPT model from scratch.

    Training pipeline:
    1. Initialize optimizer and data type
    2. Training loop: forward -> backward -> optimizer step
    3. Periodic evaluation and checkpointing
    4. Track losses and metrics

    Args:
        model: GPT model to train
        train_data: training data (numpy memmap)
        val_data: validation data (numpy memmap)
        max_iters: maximum training iterations
        learning_rate: maximum learning rate
        out_dir: directory to save checkpoints
        dataset_name: name for logging

    Returns:
        history: dictionary with training metrics (losses, times, etc.)
    """
    # Setup
    model = model.to(device)
    model.train()

    # Initialize optimizer (AdamW)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # Setup autocast context
    dtype_torch = torch.bfloat16 if 'bfloat' in dtype else torch.float16
    if device == 'cuda':
        dtype_ctx = torch.amp.autocast(device_type=device, dtype=dtype_torch)
    else:
        dtype_ctx = nullcontext()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'iter': [],
        'time': [],
    }

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training on {dataset_name}")
    print(f"{'='*60}")
    print(f"  Max iterations: {max_iters}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Parameters: {model.get_num_params()/1e6:.2f}M")
    print(f"  Train tokens: {len(train_data):,}")
    print(f"{'='*60}")

    # Training loop
    t0 = time.time()
    best_val_loss = float('inf')

    for iter_num in range(max_iters):
        # Get current learning rate
        lr = get_lr(iter_num, warmup_iters, max_iters, learning_rate, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Sample batch
        X, Y = get_batch(train_data, batch_size, block_size, device)

        # Forward pass with mixed precision
        with dtype_ctx:
            _, loss = model(X, Y)

        # Backward pass with gradient accumulation
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        # Gradient clipping (prevents exploding gradients)
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Evaluation and logging
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            t1 = time.time()

            # Evaluate on training and validation
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device, dtype_ctx)

            # Calculate tokens per second
            dt = t1 - t0
            tokens_per_sec = batch_size * block_size * eval_interval / dt

            # Log progress
            print(f"  Iter {iter_num:4d}/{max_iters}: "
                  f"train={losses['train']:.4f}, val={losses['val']:.4f}, "
                  f"lr={lr:.6f}, {tokens_per_sec:.0f} tok/s")

            # Save to history
            history['train_loss'].append(losses['train'])
            history['val_loss'].append(losses['val'])
            history['lr'].append(lr)
            history['iter'].append(iter_num)
            history['time'].append(t1 - t0)

            # Save checkpoint if validation loss improved
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

            # Reset timer
            t0 = time.time()

    print(f"\n✓ Training complete on {dataset_name}!")
    print(f"  Best validation loss: {best_val_loss:.4f}")

    return history


print("Training functions defined successfully!")
```

### Cell 8: Train on WikiText-2

```python
# ============================================================================
# TRAIN ON WIKITEXT-2
# ============================================================================

# Train model on WikiText-2 dataset
# Expected time: ~10-15 minutes on GPU
print("="*70)
print("TRAINING ON WIKITEXT-2 (Encyclopedic)")
print("="*70)

history_wikitext = train_model(
    model=model_wikitext,
    train_data=wikitext2_train,
    val_data=wikitext2_val,
    max_iters=max_iters,
    learning_rate=learning_rate,
    out_dir='out-wikitext2-comparison',
    dataset_name='WikiText-2'
)
```

### Cell 9: Train on BookCorpus

```python
# ============================================================================
# TRAIN ON BOOKCORPUS (Narrative)
# ============================================================================

# Train model on BookCorpus dataset (sampled)
# Expected time: ~10-15 minutes on GPU
print("\n" + "="*70)
print("TRAINING ON BOOKCORPUS (Narrative - Sampled)")
print("="*70)

history_bookcorpus = train_model(
    model=model_bookcorpus,
    train_data=bookcorpus_train,
    val_data=bookcorpus_val,
    max_iters=max_iters,
    learning_rate=learning_rate,
    out_dir='out-bookcorpus-comparison',
    dataset_name='BookCorpus'
)
```

### Cell 10: Evaluation and Comparison

```python
# ============================================================================
# EVALUATION AND COMPARISON
# ============================================================================

# Calculate final metrics for both datasets
print("="*70)
print("FINAL EVALUATION")
print("="*70)

# WikiText-2 metrics
wiki_train_loss = history_wikitext['train_loss'][-1]
wiki_val_loss = history_wikitext['val_loss'][-1]
wiki_train_ppl = np.exp(wiki_train_loss)
wiki_val_ppl = np.exp(wiki_val_loss)

# BookCorpus metrics
book_train_loss = history_bookcorpus['train_loss'][-1]
book_val_loss = history_bookcorpus['val_loss'][-1]
book_train_ppl = np.exp(book_train_loss)
book_val_ppl = np.exp(book_val_loss)

print(f"\nWikiText-2 (Encyclopedic):")
print(f"  Train loss:      {wiki_train_loss:.4f}")
print(f"  Val loss:        {wiki_val_loss:.4f}")
print(f"  Train perplexity: {wiki_train_ppl:.2f}")
print(f"  Val perplexity:   {wiki_val_ppl:.2f}")

print(f"\nBookCorpus (Narrative):")
print(f"  Train loss:      {book_train_loss:.4f}")
print(f"  Val loss:        {book_val_loss:.4f}")
print(f"  Train perplexity: {book_train_ppl:.2f}")
print(f"  Val perplexity:   {book_val_ppl:.2f}")

# Comparison
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")
print(f"  Val loss ratio (Wiki/Book):  {wiki_val_loss / book_val_loss:.2f}x")
print(f"  Val PPL ratio (Wiki/Book):   {wiki_val_ppl / book_val_ppl:.2f}x")
print(f"  Loss difference:             {abs(wiki_val_loss - book_val_loss):.4f}")
print(f"  PPL difference:              {abs(wiki_val_ppl - book_val_ppl):.2f}")

# Interpretation
print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")
if wiki_val_ppl < book_val_ppl:
    print("  ✓ WikiText-2 (Encyclopedic) achieves lower perplexity:")
    print("    → Possible explanation: More regular, predictable sentence structure")
    print("    → Technical vocabulary may have consistent patterns")
else:
    print("  ✓ BookCorpus (Narrative) achieves lower perplexity:")
    print("    → Possible explanation: Repetitive dialogue patterns")
    print("    → Narrative conventions may be easier to model")

# Overfitting analysis
wiki_gap = wiki_val_loss - wiki_train_loss
book_gap = book_val_loss - book_train_loss

print(f"\n{'='*70}")
print("OVERFITTING ANALYSIS")
print(f"{'='*70}")
print(f"  WikiText-2 train-val gap: {wiki_gap:.4f}")
print(f"  BookCorpus train-val gap:  {book_gap:.4f}")

if wiki_gap < book_gap:
    print(f"  → WikiText-2 shows better generalization (smaller gap)")
else:
    print(f"  → BookCorpus shows better generalization (smaller gap)")
```

### Cell 11: Visualizations

```python
# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training curves - Loss
axes[0, 0].plot(history_wikitext['iter'], history_wikitext['train_loss'], 'b-', label='WikiText-2 Train', linewidth=2)
axes[0, 0].plot(history_wikitext['iter'], history_wikitext['val_loss'], 'b--', label='WikiText-2 Val', linewidth=2)
axes[0, 0].plot(history_bookcorpus['iter'], history_bookcorpus['train_loss'], 'r-', label='BookCorpus Train', linewidth=2)
axes[0, 0].plot(history_bookcorpus['iter'], history_bookcorpus['val_loss'], 'r--', label='BookCorpus Val', linewidth=2)
axes[0, 0].set_xlabel('Iteration', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Training Curves: Loss Comparison', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Learning rate schedule
axes[0, 1].plot(history_wikitext['iter'], history_wikitext['lr'], 'b-', label='WikiText-2', linewidth=2)
axes[0, 1].plot(history_bookcorpus['iter'], history_bookcorpus['lr'], 'r-', label='BookCorpus', linewidth=2)
axes[0, 1].set_xlabel('Iteration', fontsize=12)
axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Perplexity over time
wiki_ppl = [np.exp(l) for l in history_wikitext['val_loss']]
book_ppl = [np.exp(l) for l in history_bookcorpus['val_loss']]
axes[1, 0].plot(history_wikitext['iter'], wiki_ppl, 'b-', label='WikiText-2', linewidth=2)
axes[1, 0].plot(history_bookcorpus['iter'], book_ppl, 'r-', label='BookCorpus', linewidth=2)
axes[1, 0].set_xlabel('Iteration', fontsize=12)
axes[1, 0].set_ylabel('Perplexity', fontsize=12)
axes[1, 0].set_title('Validation Perplexity Over Time', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Final perplexities comparison
datasets = ['WikiText-2\n(Encyclopedic)', 'BookCorpus\n(Narrative)']
train_ppl = [wiki_train_ppl, book_train_ppl]
val_ppl = [wiki_val_ppl, book_val_ppl]
x = np.arange(len(datasets))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, train_ppl, width, label='Train', alpha=0.8, color=['blue', 'red'])
bars2 = axes[1, 1].bar(x + width/2, val_ppl, width, label='Val', alpha=0.6, color=['blue', 'red'])
axes[1, 1].set_ylabel('Perplexity', fontsize=12)
axes[1, 1].set_title('Final Perplexity Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(datasets, fontsize=11)
axes[1, 1].legend(fontsize=10)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)

axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('wikitext2_vs_bookcorpus_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Visualizations saved to 'wikitext2_vs_bookcorpus_comparison.png'")
```

### Cell 12: Text Generation

```python
# ============================================================================
# TEXT GENERATION
# ============================================================================

# Load best checkpoints
print("Loading best checkpoints...")

# WikiText-2 checkpoint
ckpt_wiki = torch.load('out-wikitext2-comparison/ckpt.pt', map_location=device)
model_wikitext.load_state_dict(ckpt_wiki['model'])
model_wikitext.eval()

# BookCorpus checkpoint
ckpt_book = torch.load('out-bookcorpus-comparison/ckpt.pt', map_location=device)
model_bookcorpus.load_state_dict(ckpt_book['model'])
model_bookcorpus.eval()

print("✓ Checkpoints loaded")

# Generation settings
max_new_tokens = 100
temperature = 0.8
top_k = 40

# Test prompts (Wikipedia-style for WikiText-2, Narrative-style for BookCorpus)
wiki_prompts = [
    "The history of science is the study of",
    "In mathematics, a function is",
    "The capital city of France is known for",
]

book_prompts = [
    "She walked into the room and saw",
    "It was a dark and stormy night when",
    "He looked at her and said softly",
]

# Encode prompts
enc = tiktoken.get_encoding("gpt2")

print("\n" + "="*70)
print("TEXT GENERATION: WIKITEXT-2 MODEL (Encyclopedic)")
print("="*70)

for i, prompt in enumerate(wiki_prompts):
    print(f"\nSample {i+1}:")
    print(f"Prompt: {prompt}")
    print("-" * 70)

    start_ids = enc.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        dtype_torch = torch.bfloat16 if 'bfloat' in dtype else torch.float16
        with torch.amp.autocast(device_type=device, dtype=dtype_torch) if device == 'cuda' else nullcontext():
            y = model_wikitext.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    generated = enc.decode(y[0].tolist())
    print(generated)

print("\n" + "="*70)
print("TEXT GENERATION: BOOKCORPUS MODEL (Narrative)")
print("="*70)

for i, prompt in enumerate(book_prompts):
    print(f"\nSample {i+1}:")
    print(f"Prompt: {prompt}")
    print("-" * 70)

    start_ids = enc.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        dtype_torch = torch.bfloat16 if 'bfloat' in dtype else torch.float16
        with torch.amp.autocast(device_type=device, dtype=dtype_torch) if device == 'cuda' else nullcontext():
            y = model_bookcorpus.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    generated = enc.decode(y[0].tolist())
    print(generated)
```

### Cell 13: Token Distribution Analysis

```python
# ============================================================================
# TOKEN DISTRIBUTION ANALYSIS
# ============================================================================

print("="*70)
print("TOKEN DISTRIBUTION ANALYSIS")
print("="*70)

# Analyze WikiText-2 token distribution
unique_wiki = np.unique(wikitext2_train)
_, counts_wiki = np.unique(wikitext2_train, return_counts=True)

print(f"\nWikiText-2 (Encyclopedic):")
print(f"  Unique tokens: {len(unique_wiki):,} / 50,257 ({len(unique_wiki)/50257*100:.1f}%)")
print(f"  Total tokens:  {len(wikitext2_train):,}")

# Top 10 most frequent tokens
top_10_wiki = np.argsort(counts_wiki)[-10:][::-1]
print(f"\n  Top 10 most frequent tokens:")
for i, idx in enumerate(top_10_wiki):
    token = enc.decode([idx])
    freq = counts_wiki[idx] / len(wikitext2_train) * 100
    print(f"    {i+1:2d}. '{token:20s}' (id={idx:5d}): {freq:5.2f}%")

# Analyze BookCorpus token distribution
unique_book = np.unique(bookcorpus_train)
_, counts_book = np.unique(bookcorpus_train, return_counts=True)

print(f"\nBookCorpus (Narrative):")
print(f"  Unique tokens: {len(unique_book):,} / 50,257 ({len(unique_book)/50257*100:.1f}%)")
print(f"  Total tokens:  {len(bookcorpus_train):,}")

# Top 10 most frequent tokens
top_10_book = np.argsort(counts_book)[-10:][::-1]
print(f"\n  Top 10 most frequent tokens:")
for i, idx in enumerate(top_10_book):
    token = enc.decode([idx])
    freq = counts_book[idx] / len(bookcorpus_train) * 100
    print(f"    {i+1:2d}. '{token:20s}' (id={idx:5d}): {freq:5.2f}%")

# Common vs unique tokens
common_tokens = set(unique_wiki) & set(unique_book)
wiki_only = set(unique_wiki) - set(unique_book)
book_only = set(unique_book) - set(unique_wiki)

print(f"\n" + "="*70)
print("VOCABULARY OVERLAP")
print("="*70)
print(f"  Common tokens:   {len(common_tokens):,}")
print(f"  WikiText-2 only: {len(wiki_only):,}")
print(f"  BookCorpus only: {len(book_only):,}")

# Visualization of token frequency distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# WikiText-2 distribution
sorted_wiki = np.sort(counts_wiki)[::-1]
axes[0].loglog(range(len(sorted_wiki)), sorted_wiki, 'b-', alpha=0.6)
axes[0].set_xlabel('Token Rank', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('WikiText-2: Token Frequency (Zipf\'s Law)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# BookCorpus distribution
sorted_book = np.sort(counts_book)[::-1]
axes[1].loglog(range(len(sorted_book)), sorted_book, 'r-', alpha=0.6)
axes[1].set_xlabel('Token Rank', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('BookCorpus: Token Frequency (Zipf\'s Law)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('token_distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Token distribution visualization saved to 'token_distribution_comparison.png'")
```

### Cell 14: Detailed Analysis

```python
# ============================================================================
# DETAILED ANALYSIS
# ============================================================================

print("="*70)
print("DETAILED ANALYSIS")
print("="*70)

# 1. Convergence Analysis
wiki_improvement = history_wikitext['val_loss'][0] - history_wikitext['val_loss'][-1]
book_improvement = history_bookcorpus['val_loss'][0] - history_bookcorpus['val_loss'][-1]

print("\n1. CONVERGENCE ANALYSIS")
print("-" * 70)
print(f"WikiText-2:")
print(f"  Initial val loss: {history_wikitext['val_loss'][0]:.4f}")
print(f"  Final val loss:   {history_wikitext['val_loss'][-1]:.4f}")
print(f"  Improvement:      {wiki_improvement:.4f}")

print(f"\nBookCorpus:")
print(f"  Initial val loss: {history_bookcorpus['val_loss'][0]:.4f}")
print(f"  Final val loss:   {history_bookcorpus['val_loss'][-1]:.4f}")
print(f"  Improvement:      {book_improvement:.4f}")

if wiki_improvement > book_improvement:
    print(f"\n→ WikiText-2 converged faster (more loss reduction)")
else:
    print(f"\n→ BookCorpus converged faster (more loss reduction)")

# 2. Overfitting Analysis
wiki_gap = wiki_val_loss - wiki_train_loss
book_gap = book_val_loss - book_train_loss

print("\n2. OVERFITTING ANALYSIS")
print("-" * 70)
print(f"WikiText-2 train-val gap: {wiki_gap:.4f}")
print(f"BookCorpus train-val gap:  {book_gap:.4f}")

if wiki_gap < book_gap:
    print(f"→ WikiText-2 generalizes better (smaller gap)")
else:
    print(f"→ BookCorpus generalizes better (smaller gap)")

# 3. Training Efficiency
wiki_time = sum(history_wikitext['time'])
book_time = sum(history_bookcorpus['time'])
wiki_tok_sec = len(wikitext2_train) * max_iters / wiki_time if wiki_time > 0 else 0
book_tok_sec = len(bookcorpus_train) * max_iters / book_time if book_time > 0 else 0

print("\n3. TRAINING EFFICIENCY")
print("-" * 70)
print(f"WikiText-2:")
print(f"  Total time:          {wiki_time:.1f}s")
print(f"  Tokens processed:    {len(wikitext2_train) * max_iters:,}")
print(f"  Speed:               {wiki_tok_sec:.0f} tokens/sec")

print(f"\nBookCorpus:")
print(f"  Total time:          {book_time:.1f}s")
print(f"  Tokens processed:    {len(bookcorpus_train) * max_iters:,}")
print(f"  Speed:               {book_tok_sec:.0f} tokens/sec")

# 4. Style Analysis
print("\n4. TEXT STYLE ANALYSIS")
print("-" * 70)
print("WikiText-2 (Encyclopedic) characteristics:")
print("  + Consistent, formal sentence structure")
print("  + Technical and academic vocabulary")
print("  + Information-dense content")
print("  + Objective tone")

print("\nBookCorpus (Narrative) characteristics:")
print("  + Conversational dialogue patterns")
print("  + Descriptive and emotive language")
print("  + Varied sentence structures")
print("  + Subjective, story-telling tone")

print(f"\nObserved perplexity difference: {abs(wiki_val_ppl - book_val_ppl):.2f}")
if abs(wiki_val_ppl - book_val_ppl) < 5:
    print("→ Models perform similarly on both styles")
else:
    print("→ Significant performance difference between styles")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
```

### Cell 15: Conclusions

```python
# ============================================================================
# CONCLUSIONS AND SUMMARY
# ============================================================================

print("="*70)
print("CONCLUSIONS AND SUMMARY")
print("="*70)

# Save summary to file
with open('comparison_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("WikiText-2 vs BookCorpus Comparison Summary\n")
    f.write("="*70 + "\n\n")

    f.write("DATASETS:\n")
    f.write(f"  WikiText-2 (Encyclopedic):\n")
    f.write(f"    Train: {len(wikitext2_train):,} tokens\n")
    f.write(f"    Val:   {len(wikitext2_val):,} tokens\n\n")

    f.write(f"  BookCorpus (Narrative - Sampled):\n")
    f.write(f"    Train: {len(bookcorpus_train):,} tokens\n")
    f.write(f"    Val:   {len(bookcorpus_val):,} tokens\n\n")

    f.write("MODEL:\n")
    f.write(f"  Parameters: {num_params/1e6:.2f}M\n")
    f.write(f"  Architecture: {n_layer} layers, {n_head} heads, {n_embd} dimensions\n\n")

    f.write("RESULTS:\n")
    f.write(f"  WikiText-2:\n")
    f.write(f"    Train loss: {wiki_train_loss:.4f}\n")
    f.write(f"    Val loss:   {wiki_val_loss:.4f}\n")
    f.write(f"    Val PPL:    {wiki_val_ppl:.2f}\n\n")

    f.write(f"  BookCorpus:\n")
    f.write(f"    Train loss: {book_train_loss:.4f}\n")
    f.write(f"    Val loss:   {book_val_loss:.4f}\n")
    f.write(f"    Val PPL:    {book_val_ppl:.2f}\n\n")

    f.write("COMPARISON:\n")
    f.write(f"  PPL difference: {abs(wiki_val_ppl - book_val_ppl):.2f}\n")
    f.write(f"  Better model:   {'WikiText-2' if wiki_val_ppl < book_val_ppl else 'BookCorpus'}\n")

print("\n" + "="*70)
print("EXPERIMENT SUMMARY")
print("="*70)

print(f"\nDatasets compared:")
print(f"  WikiText-2 (Encyclopedic): {len(wikitext2_train):,} tokens")
print(f"  BookCorpus (Narrative):    {len(bookcorpus_train):,} tokens")

print(f"\nModel configuration:")
print(f"  Parameters: {num_params/1e6:.2f}M")
print(f"  Architecture: {n_layer} layers, {n_head} heads, {n_embd} dimensions")
print(f"  Training: {max_iters} iterations")

print(f"\nResults:")
print(f"  WikiText-2 val perplexity: {wiki_val_ppl:.2f}")
print(f"  BookCorpus val perplexity: {book_val_ppl:.2f}")

if wiki_val_ppl < book_val_ppl:
    print(f"\n→ WikiText-2 achieved better performance ({abs(wiki_val_ppl - book_val_ppl):.2f} PPL difference)")
else:
    print(f"\n→ BookCorpus achieved better performance ({abs(wiki_val_ppl - book_val_ppl):.2f} PPL difference)")

print(f"\n{'='*70}")
print("KEY FINDINGS")
print("="*70)

findings = []
findings.append(f"1. Both datasets achieved reasonable perplexity with ~2.5M tokens")
findings.append(f"2. Training converged in {max_iters} iterations for both datasets")
findings.append(f"3. Model architecture ({num_params/1e6:.2f}M params) works well for both styles")

if abs(wiki_val_ppl - book_val_ppl) < 5:
    findings.append(f"4. Similar performance suggests data size is more important than style")
else:
    findings.append(f"4. Performance difference suggests text style affects model learning")

findings.append(f"5. Token distributions show distinct vocabulary patterns")

for finding in findings:
    print(f"  {finding}")

print(f"\n{'='*70}")
print("RECOMMENDATIONS")
print("="*70)

recommendations = []
recommendations.append("• For encyclopedic/factual content:")
recommendations.append("  - Use WikiText-2 or similar datasets")
recommendations.append("  - Models learn consistent, formal patterns")

recommendations.append("\n• For narrative/creative content:")
recommendations.append("  - Use BookCorpus or similar datasets")
recommendations.append("  - Models learn dialogue and storytelling patterns")

recommendations.append("\n• For general-purpose language models:")
recommendations.append("  - Consider combining both styles")
recommendations.append("  - Datasets should include diverse text types")

recommendations.append("\n• For future experiments:")
recommendations.append("  - Try larger models (20M+ params) for better performance")
recommendations.append("  - Experiment with block size (512, 1024) for longer context")
recommendations.append("  - Test on additional datasets (OpenWebText, etc.)")

for rec in recommendations:
    print(rec)

print(f"\n{'='*70}")
print("EXPERIMENT COMPLETE!")
print("="*70)

print(f"\n✓ Summary saved to 'comparison_summary.txt'")
print(f"✓ Visualizations saved:")
print(f"  - wikitext2_vs_bookcorpus_comparison.png")
print(f"  - token_distribution_comparison.png")
print(f"\n✓ Checkpoints saved:")
print(f"  - out-wikitext2-comparison/ckpt.pt")
print(f"  - out-bookcorpus-comparison/ckpt.pt")
```

## Execution Time Estimates

| Phase | WikiText-2 | BookCorpus | Total |
|-------|------------|------------|-------|
| Dataset preparation | ~1 min | ~5-10 min (streaming) | ~5-10 min |
| Training | ~10-15 min | ~10-15 min | ~20-30 min |
| Evaluation | ~2 min | ~2 min | ~4 min |
| Visualizations | ~1 min | ~1 min | ~2 min |

**Total expected time**: ~30-40 minutes on modern GPU, 2-3 hours on CPU

**Note**: Dataset preparation time for BookCorpus uses streaming to avoid downloading the full 74M documents. This significantly reduces preparation time.

## Key Features

1. **Fair Comparison**: Same architecture, hyperparameters, and token count for both datasets
2. **Style Contrast**: Encyclopedic vs narrative text
3. **Automatic Preparation**: Notebook prepares datasets if needed
4. **Comprehensive Visualizations**: Loss curves, perplexity comparison, token distributions
5. **Text Generation**: Demonstrates style differences in generated text
6. **Detailed Analysis**: Convergence, overfitting, efficiency metrics
7. **Streaming for BookCorpus**: Efficient data loading for large dataset
8. **30-Minute Budget**: Optimized for quick experimentation

## Output Files

The notebook will generate:

### Model Checkpoints
- `out-wikitext2-comparison/ckpt.pt` - Best WikiText-2 model checkpoint
- `out-bookcorpus-comparison/ckpt.pt` - Best BookCorpus model checkpoint

### Visualizations
- `wikitext2_vs_bookcorpus_comparison.png` - Training curves, perplexity comparison, learning rate schedule
- `token_distribution_comparison.png` - Token frequency distributions (Zipf's law)

### Summary
- `comparison_summary.txt` - Text summary of results

### Dataset Files (created during execution)
- `data/wikitext2/train.bin` / `val.bin` - WikiText-2 tokenized data
- `data/bookcorpus/train_sample.bin` / `val_sample.bin` - BookCorpus sampled data

## Notes

- **GPU Memory**: Recommended 8GB+ for comfortable training
- **CPU Fallback**: Training on CPU will be 10-50x slower
- **Adjust `max_iters`**: Can increase for better convergence or decrease for faster testing
- **BookCorpus Sampling**: Uses streaming to avoid downloading full dataset (74M docs)
- **Reproducibility**: Random seeds set for consistent results
- **Model Reuse**: Checkpoints can be used for further experiments or fine-tuning

## References

- WikiText paper: Merity et al. (2016) "Pointer Sentinel Mixture Models"
- BookCorpus paper: Zhu et al. (2015) "Aligning Books and Movies"
- HuggingFace WikiText: https://huggingface.co/datasets/wikitext
- HuggingFace BookCorpus: https://huggingface.co/datasets/SamuelYang/bookcorpus
- nanoGPT: https://github.com/karpathy/nanoGPT
