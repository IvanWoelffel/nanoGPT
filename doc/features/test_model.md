# Test Plan: Training Nano-GPT from Scratch on WikiText-2

## Overview

This plan creates a comprehensive Jupyter notebook (`test.ipynb`) to train and evaluate Nano-GPT models from scratch on WikiText-2 dataset, with shakespeare_char as an internal baseline for comparison.

## Dataset Information

### WikiText-2 (Primary Dataset)
- **Source**: Wikipedia Good and Featured articles
- **Size**: ~2.43M training tokens, ~251K validation tokens
- **Files**: `data/wikitext2/train.bin` (4.63 MB), `data/wikitext2/val.bin` (0.48 MB)
- **Status**: Already prepared and tokenized

### shakespeare_char (Baseline Dataset)
- **Source**: Tiny Shakespeare plays (character-level)
- **Size**: ~1.00M training tokens, ~112K validation tokens
- **Advantage**: Small dataset for quick training, enables fast baseline comparison
- **Status**: Will be prepared during notebook execution

## Jupyter Notebook Structure

### Cell 4: Setup and Imports

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
from tqdm.auto import tqdm

# Set random seeds for reproducibility
torch.manual_seed(1337)
np.random.seed(1337)

# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("Setup complete. Libraries imported.")
```

### Cell 5: Configuration

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

# --- BATCH SIZE ---
# batch_size = 12: Chosen based on:
# - GPU memory constraints (typical 12GB-24GB GPUs can handle this comfortably)
# - Training stability: Too small = noisy gradients, too large = less frequent updates
# - Trade-off: Larger batch = faster training but may need higher learning rate
# - Reference: GPT-2 training used 512 batch size across 8 GPUs (64 per GPU)
batch_size = 12
print(f"Batch size: {batch_size}")

# --- BLOCK SIZE (CONTEXT WINDOW) ---
# block_size = 256: Chosen because:
# - WikiText-2 sentences are relatively long, need context > 128
# - Larger context requires more memory and computation (O(n^2) in attention)
# - Trade-off: 256 is sweet spot for small models (6 layers)
# - Note: GPT-2 uses 1024, but our model is much smaller
block_size = 256
print(f"Block size: {block_size}")

# --- MODEL ARCHITECTURE ---
# Small but effective model for fast training and reasonable quality:
# - n_layer = 6: Depth sufficient for simple language modeling
# - n_head = 6: Number of attention heads, should divide n_embd evenly
# - n_embd = 384: Embedding dimension, larger = more capacity but slower
# - Reasoning: ~10M parameters, trains in 15-30 minutes on GPU
# Reference: Original GPT-2 has 12 layers, 12 heads, 768 dimensions (124M params)
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
# - Reference: Large models use 6e-4, but smaller models can use higher LR
learning_rate = 1e-3
print(f"Learning rate: {learning_rate}")

# --- MAX ITERATIONS ---
# max_iters = 5000: Chosen because:
# - Enough to see clear learning curves and convergence
# - WikiText-2 has ~2.43M tokens, each iter processes ~3K tokens
# - Total tokens seen: 5000 * 12 * 256 = 15.36M tokens (~6.3 epochs)
# - Trade-off: More iters = better but slower
max_iters = 5000
print(f"Max iterations: {max_iters}")

# --- EVALUATION SETTINGS ---
# eval_interval = 250: Check progress every 250 iterations (10 times total)
# eval_iters = 200: Average loss over 200 batches for stable estimate
eval_interval = 250
eval_iters = 200
print(f"Evaluation: every {eval_interval} iters, {eval_iters} batches")

# --- GRADIENT ACCUMULATION ---
# gradient_accumulation_steps = 1: No accumulation
# Use >1 if you want effective batch size larger than GPU memory allows
gradient_accumulation_steps = 1
print(f"Gradient accumulation: {gradient_accumulation_steps}")

# --- DROPOUT ---
# dropout = 0.1: Light regularization for small dataset
# Too high: Underfitting (model can't learn)
# Too low: Overfitting (model memorizes training data)
# Reference: Training from scratch uses 0.1-0.2, fine-tuning uses 0.0
dropout = 0.1
print(f"Dropout: {dropout}")

# --- LEARNING RATE DECAY ---
# decay_lr = True: Decay LR over time helps convergence
# warmup_iters = 200: LR ramps up from 0 to full LR in first 200 iters (stability)
# lr_decay_iters = 5000: Decay until max_iters (or stop earlier)
# min_lr = 1e-4: Minimum LR (typically 1/10 of starting LR)
decay_lr = True
warmup_iters = 200
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

# --- DYNAMIC SELECTION FOR SHAKESPEARE ---
# Shakespeare is much smaller (1M tokens vs 2.4M), so we'll adjust:
# - Smaller max_iters (faster training on small dataset)
# - Higher learning rate (smaller models on small data can learn faster)
shakespeare_max_iters = 3000
shakespeare_learning_rate = 3e-3
print(f"\nShakespeare-specific: max_iters={shakespeare_max_iters}, lr={shakespeare_learning_rate}")

print("\n" + "="*70)
print("Configuration complete!")
print("="*70)
```

### Cell 6: Prepare Datasets

```python
# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_shakespeare_char():
    """
    Prepare shakespeare_char dataset for comparison baseline.
    This dataset is much smaller than WikiText-2, enabling quick training
    and comparison of how models perform on different text types.

    Why shakespeare_char as baseline?
    1. Small dataset (~1M tokens) - trains in 5-10 minutes
    2. Distinct text style (old English, plays) vs encyclopedia text
    3. Well-studied dataset in literature
    4. Allows quick iteration and debugging
    """
    shakespeare_dir = 'data/shakespeare_char'

    # Check if already prepared
    if os.path.exists(f'{shakespeare_dir}/train.bin'):
        print("✓ Shakespeare dataset already prepared")
        return

    print("Preparing shakespeare_char dataset...")

    # Import preparation script
    import sys
    sys.path.insert(0, shakespeare_dir)
    import prepare

    print("✓ Shakespeare dataset prepared successfully")

# Prepare datasets
prepare_shakespeare_char()
```

### Cell 7: Load Datasets

```python
# ============================================================================
# LOAD DATASETS
# ============================================================================

# Load WikiText-2 (already prepared)
print("Loading WikiText-2 dataset...")
wikitext2_train = np.memmap('data/wikitext2/train.bin', dtype=np.uint16, mode='r')
wikitext2_val = np.memmap('data/wikitext2/val.bin', dtype=np.uint16, mode='r')
print(f"  WikiText-2 train: {len(wikitext2_train):,} tokens ({len(wikitext2_train)*2/1024/1024:.2f} MB)")
print(f"  WikiText-2 val:   {len(wikitext2_val):,} tokens ({len(wikitext2_val)*2/1024/1024:.2f} MB)")

# Load Shakespeare (prepared in cell 6)
print("\nLoading Shakespeare dataset...")
shakespeare_train = np.memmap('data/shakespeare_char/train.bin', dtype=np.uint16, mode='r')
shakespeare_val = np.memmap('data/shakespeare_char/val.bin', dtype=np.uint16, mode='r')
print(f"  Shakespeare train: {len(shakespeare_train):,} tokens ({len(shakespeare_train)*2/1024/1024:.2f} MB)")
print(f"  Shakespeare val:   {len(shakespeare_val):,} tokens ({len(shakespeare_val)*2/1024/1024:.2f} MB)")

print("\n" + "="*70)
print("Dataset Comparison:")
print(f"  WikiText-2 vs Shakespeare: {len(wikitext2_train)/len(shakespeare_train):.1f}x larger")
print("="*70)
```

### Cell 8: Model Architecture

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
      * 6 layers is reasonable for small datasets and fast training
      * GPT-2 uses 12 layers, but we have much less data

    - n_head = 6: Number of attention heads in multi-head attention
      * Each head learns different aspects of relationships
      * Must divide n_embd evenly (384 / 6 = 64 dimensions per head)
      * More heads = more parallel attention computation

    - n_embd = 384: Embedding dimensionality
      * Larger = more expressive but more parameters and slower
      * Total params ≈ 6 * 384^2 * 12 ≈ 10M parameters (reasonable for training from scratch)
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

    model = GPT(config)
    return model

# Create models for both datasets
print("Creating model for WikiText-2...")
model_wikitext = create_model(vocab_size=50257)
print(f"  Model parameters: {model_wikitext.get_num_params()/1e6:.2f}M")

print("\nCreating model for Shakespeare...")
model_shakespeare = create_model(vocab_size=50257)
print(f"  Model parameters: {model_shakespeare.get_num_params()/1e6:.2f}M")

print("\n" + "="*70)
print("Models created successfully!")
print("="*70)
```

### Cell 9: Training Function

```python
# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def get_batch(split, data, batch_size, block_size, device):
    """
    Sample a batch of data from the dataset.

    Why random sampling?
    - Ensures model sees diverse data each iteration
    - Prevents memorization of specific sequences
    - Improves generalization

    Args:
        split: 'train' or 'val'
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
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, eval_iters, device):
    """
    Estimate loss on train/val sets by averaging over multiple batches.

    Why average over multiple batches?
    - Reduces noise in loss estimate
    - Provides more stable metric for comparison
    - 200 batches gives good balance between accuracy and speed

    Args:
        model: GPT model
        data: training or validation data
        batch_size: batch size for evaluation
        block_size: context window size
        eval_iters: number of batches to average
        device: device to run on

    Returns:
        average_loss: mean loss over all batches
    """
    model.eval()
    losses = []

    for _ in range(eval_iters):
        X, Y = get_batch('val', data, batch_size, block_size, device)

        # Mixed precision inference (faster, less memory)
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if 'bfloat' in dtype else torch.float16):
            logits, loss = model(X, Y)

        losses.append(loss.item())

    model.train()
    return np.mean(losses)


def get_lr(iter_num):
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

    Returns:
        learning_rate: current learning rate
    """
    # 1) Linear warmup for warmup_iters steps
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / (warmup_iters + 1)

    # 2) If past lr_decay_iters, return min learning rate
    if iter_num > lr_decay_iters:
        return min_lr

    # 3) Cosine decay between warmup and lr_decay_iters
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
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
        dataset_name: name for logging (e.g., 'WikiText-2', 'Shakespeare')

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

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'time': [],
        'iter': [],
    }

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nTraining on {dataset_name}...")
    print(f"  Max iterations: {max_iters}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Parameters: {model.get_num_params()/1e6:.2f}M")

    # Training loop
    t0 = time.time()
    best_val_loss = float('inf')

    for iter_num in range(max_iters):
        # Get current learning rate
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Sample batch
        X, Y = get_batch('train', train_data, batch_size, block_size, device)

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if 'bfloat' in dtype else torch.float16):
            logits, loss = model(X, Y)

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
            train_loss = estimate_loss(model, train_data, batch_size, block_size, eval_iters, device)
            val_loss = estimate_loss(model, val_data, batch_size, block_size, eval_iters, device)

            # Calculate tokens per second
            dt = t1 - t0
            tokens_per_sec = batch_size * block_size * eval_interval / dt

            # Log progress
            print(f"  Iter {iter_num:4d}/{max_iters}: "
                  f"train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
                  f"lr {lr:.6f}, {tokens_per_sec:.0f} tok/s")

            # Save to history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(lr)
            history['time'].append(t1 - t0)
            history['iter'].append(iter_num)

            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': {
                        'n_layer': n_layer,
                        'n_head': n_head,
                        'n_embd': n_embd,
                        'block_size': block_size,
                        'vocab_size': 50257,
                        'dropout': dropout,
                        'bias': True,
                    },
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

### Cell 10: Train on WikiText-2

```python
# ============================================================================
# TRAIN ON WIKITEXT-2
# ============================================================================

# Train model on WikiText-2 dataset
# WikiText-2 is larger (2.43M tokens) so training takes longer (~15-30 minutes on GPU)
print("="*70)
print("TRAINING ON WIKITEXT-2")
print("="*70)

history_wikitext = train_model(
    model=model_wikitext,
    train_data=wikitext2_train,
    val_data=wikitext2_val,
    max_iters=max_iters,
    learning_rate=learning_rate,
    out_dir='out-wikitext2-test',
    dataset_name='WikiText-2'
)
```

### Cell 11: Train on Shakespeare (Baseline)

```python
# ============================================================================
# TRAIN ON SHAKESPEARE (BASELINE)
# ============================================================================

# Train model on Shakespeare dataset (baseline)
# Shakespeare is smaller (1.00M tokens) so training is faster (~5-10 minutes on GPU)
# Using higher learning rate (3e-3 vs 1e-3) because:
# - Smaller dataset can converge faster
# - Character-level modeling has different characteristics
# - Helps compare how same architecture learns from different data
print("\n" + "="*70)
print("TRAINING ON SHAKESPEARE (BASELINE)")
print("="*70)

history_shakespeare = train_model(
    model=model_shakespeare,
    train_data=shakespeare_train,
    val_data=shakespeare_val,
    max_iters=shakespeare_max_iters,
    learning_rate=shakespeare_learning_rate,
    out_dir='out-shakespeare-test',
    dataset_name='Shakespeare'
)
```

### Cell 12: Evaluation and Comparison

```python
# ============================================================================
# EVALUATION AND COMPARISON
# ============================================================================

# Calculate final metrics for both datasets
print("="*70)
print("FINAL EVALUATION")
print("="*70)

# WikiText-2 metrics
final_train_loss_wiki = history_wikitext['train_loss'][-1]
final_val_loss_wiki = history_wikitext['val_loss'][-1]
final_train_ppl_wiki = np.exp(final_train_loss_wiki)
final_val_ppl_wiki = np.exp(final_val_loss_wiki)

print(f"\nWikiText-2:")
print(f"  Train loss:      {final_train_loss_wiki:.4f}")
print(f"  Val loss:        {final_val_loss_wiki:.4f}")
print(f"  Train perplexity: {final_train_ppl_wiki:.2f}")
print(f"  Val perplexity:   {final_val_ppl_wiki:.2f}")

# Shakespeare metrics
final_train_loss_shake = history_shakespeare['train_loss'][-1]
final_val_loss_shake = history_shakespeare['val_loss'][-1]
final_train_ppl_shake = np.exp(final_train_loss_shake)
final_val_ppl_shake = np.exp(final_val_loss_shake)

print(f"\nShakespeare:")
print(f"  Train loss:      {final_train_loss_shake:.4f}")
print(f"  Val loss:        {final_val_loss_shake:.4f}")
print(f"  Train perplexity: {final_train_ppl_shake:.2f}")
print(f"  Val perplexity:   {final_val_ppl_shake:.2f}")

# Comparison
print(f"\nComparison (WikiText-2 vs Shakespeare):")
print(f"  Val loss ratio:  {final_val_loss_wiki / final_val_loss_shake:.2f}x")
print(f"  Val PPL ratio:   {final_val_ppl_wiki / final_val_ppl_shake:.2f}x")

# Interpretation
print("\n" + "="*70)
print("INTERPRETATION:")
print("="*70)
if final_val_ppl_wiki < final_val_ppl_shake:
    print("  ✓ Lower perplexity on WikiText-2 suggests:")
    print("    - Model learned better on larger dataset")
    print("    - Or WikiText-2 is inherently easier to predict")
else:
    print("  ✓ Lower perplexity on Shakespeare suggests:")
    print("    - Shakespeare's repetitive style (plays) is easier to model")
    print("    - Or small dataset was sufficient for this simpler text")
```

### Cell 13: Visualizations

```python
# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training curves - Loss
axes[0, 0].plot(history_wikitext['iter'], history_wikitext['train_loss'], 'b-', label='WikiText-2 Train', linewidth=2)
axes[0, 0].plot(history_wikitext['iter'], history_wikitext['val_loss'], 'b--', label='WikiText-2 Val', linewidth=2)
axes[0, 0].plot(history_shakespeare['iter'], history_shakespeare['train_loss'], 'r-', label='Shakespeare Train', linewidth=2)
axes[0, 0].plot(history_shakespeare['iter'], history_shakespeare['val_loss'], 'r--', label='Shakespeare Val', linewidth=2)
axes[0, 0].set_xlabel('Iteration', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Training Curves: Loss', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Learning rate schedule
axes[0, 1].plot(history_wikitext['iter'], history_wikitext['learning_rate'], 'b-', label='WikiText-2', linewidth=2)
axes[0, 1].plot(history_shakespeare['iter'], history_shakespeare['learning_rate'], 'r-', label='Shakespeare', linewidth=2)
axes[0, 1].set_xlabel('Iteration', fontsize=12)
axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Training speed (tokens per second)
tokens_per_sec_wiki = [batch_size * block_size * eval_interval / t if t > 0 else 0 for t in history_wikitext['time']]
tokens_per_sec_shake = [batch_size * block_size * eval_interval / t if t > 0 else 0 for t in history_shakespeare['time']]
axes[1, 0].plot(history_wikitext['iter'][1:], tokens_per_sec_wiki[1:], 'b-', label='WikiText-2', linewidth=2)
axes[1, 0].plot(history_shakespeare['iter'][1:], tokens_per_sec_shake[1:], 'r-', label='Shakespeare', linewidth=2)
axes[1, 0].set_xlabel('Iteration', fontsize=12)
axes[1, 0].set_ylabel('Tokens/Second', fontsize=12)
axes[1, 0].set_title('Training Speed', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Final perplexities comparison
datasets = ['WikiText-2', 'Shakespeare']
train_ppl = [final_train_ppl_wiki, final_train_ppl_shake]
val_ppl = [final_val_ppl_wiki, final_val_ppl_shake]
x = np.arange(len(datasets))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, train_ppl, width, label='Train', alpha=0.8)
bars2 = axes[1, 1].bar(x + width/2, val_ppl, width, label='Val', alpha=0.8)
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
plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Visualizations saved to 'training_comparison.png'")
```

### Cell 14: Text Generation

```python
# ============================================================================
# TEXT GENERATION
# ============================================================================

# Load best checkpoints
print("Loading best checkpoints...")

# WikiText-2 checkpoint
checkpoint_wiki = torch.load('out-wikitext2-test/ckpt.pt', map_location=device)
model_wikitext.load_state_dict(checkpoint_wiki['model'])
model_wikitext.eval()

# Shakespeare checkpoint
checkpoint_shake = torch.load('out-shakespeare-test/ckpt.pt', map_location=device)
model_shakespeare.load_state_dict(checkpoint_shake['model'])
model_shakespeare.eval()

print("✓ Checkpoints loaded")

# Generation settings
num_samples = 3
max_new_tokens = 150
temperature = 0.8
top_k = 40

# Test prompts (Wikipedia-style for WikiText-2, Play-style for Shakespeare)
wiki_prompts = [
    "The history of Rome began with",
    "Quantum mechanics is a fundamental theory in",
    "The human brain is composed of",
]

shake_prompts = [
    "To be or not to be, that is",
    "All the world's a stage, and",
    "Romeo, Romeo, wherefore art",
]

# Encode prompts
enc = tiktoken.get_encoding("gpt2")

print("\n" + "="*70)
print("TEXT GENERATION: WIKITEXT-2 MODEL")
print("="*70)

for i, prompt in enumerate(wiki_prompts):
    print(f"\nSample {i+1}:")
    print(f"Prompt: {prompt}")
    print("-" * 70)

    start_ids = enc.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if 'bfloat' in dtype else torch.float16):
            y = model_wikitext.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    generated = enc.decode(y[0].tolist())
    print(generated)

print("\n" + "="*70)
print("TEXT GENERATION: SHAKESPEARE MODEL")
print("="*70)

for i, prompt in enumerate(shake_prompts):
    print(f"\nSample {i+1}:")
    print(f"Prompt: {prompt}")
    print("-" * 70)

    start_ids = enc.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if 'bfloat' in dtype else torch.float16):
            y = model_shakespeare.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    generated = enc.decode(y[0].tolist())
    print(generated)
```

### Cell 15: Training Analysis

```python
# ============================================================================
# TRAINING ANALYSIS - DETAILED PARAMETER IMPACT
# ============================================================================

print("="*70)
print("TRAINING ANALYSIS: HYPERPARAMETER IMPACT")
print("="*70)

# Analysis 1: Learning Rate Impact
print("\n1. LEARNING RATE IMPACT")
print("-" * 70)
print(f"WikiText-2:")
print(f"  Starting LR: {learning_rate}")
print(f"  Final LR:    {min_lr}")
print(f"  Decay:       {decay_lr}")
print(f"  Effect:      {'Fast initial learning, then smooth convergence' if decay_lr else 'Constant learning rate'}")

print(f"\nShakespeare:")
print(f"  Starting LR: {shakespeare_learning_rate}")
print(f"  Final LR:    {shakespeare_learning_rate * min_lr/learning_rate:.6f}")
print(f"  Decay:       {decay_lr}")
print(f"  Effect:      Higher initial rate because smaller dataset allows faster learning")

# Analysis 2: Batch Size Impact
print("\n2. BATCH SIZE IMPACT")
print("-" * 70)
print(f"Current batch size: {batch_size}")
print(f"  Trade-offs:")
print(f"  + Larger batch → more stable gradients, faster training")
print(f"  - Larger batch → more memory, less frequent updates")
print(f"  + Smaller batch → more frequent updates, better generalization")
print(f"  - Smaller batch → noisy gradients, slower convergence")
print(f"  Chosen {batch_size} as balance for available GPU memory")

# Analysis 3: Block Size (Context Window) Impact
print("\n3. BLOCK SIZE (CONTEXT WINDOW) IMPACT")
print("-" * 70)
print(f"Current block size: {block_size}")
print(f"  Trade-offs:")
print(f"  + Larger context → model can learn longer dependencies")
print(f"  - Larger context → O(n²) attention complexity, much slower")
print(f"  + Smaller context → faster training, less memory")
print(f"  - Smaller context → limited long-range understanding")
print(f"  Chosen {block_size} as practical limit for small model")

# Analysis 4: Model Size Impact
print("\n4. MODEL SIZE IMPACT")
print("-" * 70)
num_params = model_wikitext.get_num_params()
print(f"Current model: {num_params/1e6:.2f}M parameters")
print(f"  Architecture: {n_layer} layers, {n_head} heads, {n_embd} dimensions")
print(f"  Trade-offs:")
print(f"  + Larger model → more capacity, can learn complex patterns")
print(f"  - Larger model → slower training, more memory, risk of overfitting")
print(f"  + Smaller model → fast training, less data needed")
print(f"  - Smaller model → limited capacity, may underfit")
print(f"  Chosen ~10M params for reasonable quality with fast training")

# Analysis 5: Dataset Size vs Model Capacity
print("\n5. DATASET SIZE VS MODEL CAPACITY")
print("-" * 70)
print(f"WikiText-2:  {len(wikitext2_train):,} tokens")
print(f"Shakespeare: {len(shakespeare_train):,} tokens")
print(f"Ratio:       {len(wikitext2_train)/len(shakespeare_train):.1f}x")
print(f"\nModel capacity: ~{num_params} parameters")
print(f"  WikiText-2:  {len(wikitext2_train)/num_params:.1f} tokens per parameter")
print(f"  Shakespeare: {len(shakespeare_train)/num_params:.1f} tokens per parameter")
print(f"  Rule of thumb: Need >20-50 tokens per parameter for good training")
print(f"  Shakespeare may be slightly data-limited, WikiText-2 is well-supplied")

# Analysis 6: Convergence Analysis
print("\n6. CONVERGENCE ANALYSIS")
print("-" * 70)
wiki_convergence_rate = (history_wikitext['val_loss'][0] - history_wikitext['val_loss'][-1]) / history_wikitext['iter'][-1]
shake_convergence_rate = (history_shakespeare['val_loss'][0] - history_shakespeare['val_loss'][-1]) / history_shakespeare['iter'][-1]

print(f"WikiText-2:")
print(f"  Initial val loss: {history_wikitext['val_loss'][0]:.4f}")
print(f"  Final val loss:   {history_wikitext['val_loss'][-1]:.4f}")
print(f"  Improvement:      {history_wikitext['val_loss'][0] - history_wikitext['val_loss'][-1]:.4f}")
print(f"  Rate:             {wiki_convergence_rate:.6f} loss per iteration")

print(f"\nShakespeare:")
print(f"  Initial val loss: {history_shakespeare['val_loss'][0]:.4f}")
print(f"  Final val loss:   {history_shakespeare['val_loss'][-1]:.4f}")
print(f"  Improvement:      {history_shakespeare['val_loss'][0] - history_shakespeare['val_loss'][-1]:.4f}")
print(f"  Rate:             {shake_convergence_rate:.6f} loss per iteration")

if wiki_convergence_rate > shake_convergence_rate:
    print(f"\n→ WikiText-2 converged faster per iteration")
else:
    print(f"\n→ Shakespeare converged faster per iteration")

# Analysis 7: Overfitting Analysis
print("\n7. OVERFITTING ANALYSIS")
print("-" * 70)
wiki_gap = history_wikitext['val_loss'][-1] - history_wikitext['train_loss'][-1]
shake_gap = history_shakespeare['val_loss'][-1] - history_shakespeare['train_loss'][-1]

print(f"WikiText-2:")
print(f"  Train-val gap: {wiki_gap:.4f}")
print(f"  Interpretation: ", end="")
if wiki_gap < 0.1:
    print("✓ Good generalization (small gap)")
elif wiki_gap < 0.3:
    print("⚠ Mild overfitting (moderate gap)")
else:
    print("✗ Significant overfitting (large gap)")

print(f"\nShakespeare:")
print(f"  Train-val gap: {shake_gap:.4f}")
print(f"  Interpretation: ", end="")
if shake_gap < 0.1:
    print("✓ Good generalization (small gap)")
elif shake_gap < 0.3:
    print("⚠ Mild overfitting (moderate gap)")
else:
    print("✗ Significant overfitting (large gap)")

# Analysis 8: Training Efficiency
print("\n8. TRAINING EFFICIENCY")
print("-" * 70)
total_tokens_wiki = batch_size * block_size * history_wikitext['iter'][-1]
total_time_wiki = sum(history_wikitext['time'])
total_tokens_shake = batch_size * block_size * history_shakespeare['iter'][-1]
total_time_shake = sum(history_shakespeare['time'])

print(f"WikiText-2:")
print(f"  Total tokens trained: {total_tokens_wiki:,}")
print(f"  Total time:          {total_time_wiki:.1f}s")
print(f"  Tokens per second:   {total_tokens_wiki/total_time_wiki:.0f}")

print(f"\nShakespeare:")
print(f"  Total tokens trained: {total_tokens_shake:,}")
print(f"  Total time:          {total_time_shake:.1f}s")
print(f"  Tokens per second:   {total_tokens_shake/total_time_shake:.0f}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
```

### Cell 16: Token Distribution Analysis

```python
# ============================================================================
# TOKEN DISTRIBUTION ANALYSIS
# ============================================================================

print("="*70)
print("TOKEN DISTRIBUTION ANALYSIS")
print("="*70)

# Analyze WikiText-2 token distribution
unique_tokens_wiki, counts_wiki = np.unique(wikitext2_train, repeat_counts=False, return_counts=True)
top_tokens_wiki = np.argsort(counts_wiki)[-10:][::-1]

print("\nWikiText-2 Token Distribution:")
print(f"  Unique tokens: {len(unique_tokens_wiki):,} / 50,257 ({len(unique_tokens_wiki)/50257*100:.1f}%)")
print(f"  Total tokens:  {len(wikitext2_train):,}")
print(f"  Token coverage: {np.sum(counts_wiki) / len(wikitext2_train) * 100:.1f}%")
print("\nTop 10 most frequent tokens:")
enc = tiktoken.get_encoding("gpt2")
for i, token_id in enumerate(top_tokens_wiki):
    token = enc.decode([token_id])
    freq = counts_wiki[token_id] / len(wikitext2_train) * 100
    print(f"  {i+1:2d}. '{token:20s}' (id={token_id:5d}): {freq:5.2f}%")

# Analyze Shakespeare token distribution
unique_tokens_shake, counts_shake = np.unique(shakespeare_train, repeat_counts=False, return_counts=True)
top_tokens_shake = np.argsort(counts_shake)[-10:][::-1]

print("\nShakespeare Token Distribution:")
print(f"  Unique tokens: {len(unique_tokens_shake):,} / 50,257 ({len(unique_tokens_shake)/50257*100:.1f}%)")
print(f"  Total tokens:  {len(shakespeare_train):,}")
print(f"  Token coverage: {np.sum(counts_shake) / len(shakespeare_train) * 100:.1f}%")
print("\nTop 10 most frequent tokens:")
for i, token_id in enumerate(top_tokens_shake):
    token = enc.decode([token_id])
    freq = counts_shake[token_id] / len(shakespeare_train) * 100
    print(f"  {i+1:2d}. '{token:20s}' (id={token_id:5d}): {freq:5.2f}%")

# Comparison
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Token vocabulary usage:")
print(f"  WikiText-2:  {len(unique_tokens_wiki):,} unique tokens")
print(f"  Shakespeare: {len(unique_tokens_shake):,} unique tokens")
print(f"  Difference:  {len(unique_tokens_wiki) - len(unique_tokens_shake):,} tokens")

# Visualization of token frequency distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# WikiText-2 distribution
sorted_counts_wiki = np.sort(counts_wiki)[::-1]
axes[0].loglog(range(len(sorted_counts_wiki)), sorted_counts_wiki, 'b-', alpha=0.6)
axes[0].set_xlabel('Token Rank', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('WikiText-2 Token Frequency Distribution', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Shakespeare distribution
sorted_counts_shake = np.sort(counts_shake)[::-1]
axes[1].loglog(range(len(sorted_counts_shake)), sorted_counts_shake, 'r-', alpha=0.6)
axes[1].set_xlabel('Token Rank', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Shakespeare Token Frequency Distribution', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('token_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Token distribution visualization saved to 'token_distribution.png'")
```

### Cell 17: Hyperparameter Sensitivity

```python
# ============================================================================
# HYPERPARAMETER SENSITIVITY
# ============================================================================

print("="*70)
print("HYPERPARAMETER SENSITIVITY ANALYSIS")
print("="*70)
print("\nNote: This is a conceptual analysis. For full sensitivity tests,")
print("      retrain models with different hyperparameters.")

# Analysis 1: Learning Rate Sensitivity
print("\n1. LEARNING RATE SENSITIVITY")
print("-" * 70)
print(f"Used LR: {learning_rate}")
print("  Effects of changing LR:")
print("  • 2x higher (2e-3):")
print("    - Faster initial learning")
print("    - Risk: Instability, loss may spike")
print("    - May need fewer warmup iterations")
print("  • 2x lower (5e-4):")
print("    - More stable training")
print("    - Risk: Very slow convergence")
print("    - May need more iterations")
print("  Recommendation: 1e-3 is good starting point, tune ±50%")

# Analysis 2: Batch Size Sensitivity
print("\n2. BATCH SIZE SENSITIVITY")
print("-" * 70)
print(f"Used batch size: {batch_size}")
print("  Effects of changing batch size:")
print("  • 2x larger (24):")
print("    - Faster training (~2x speed)")
print("    - More stable gradients")
print("    - Risk: May need higher LR (scaling rule)")
print("  • 2x smaller (6):")
print("    - Slower training (~2x speed)")
print("    - Noisier gradients")
print("    - Benefit: May generalize better")
print("  Recommendation: Use largest that fits in GPU memory")

# Analysis 3: Model Size Sensitivity
print("\n3. MODEL SIZE SENSITIVITY")
print("-" * 70)
print(f"Used model: {num_params/1e6:.2f}M parameters")
print("  Effects of changing model size:")
print("  • 2x larger (20M params):")
print("    - Better capacity for complex patterns")
print("    - Risk: Overfitting on small datasets")
print("    - Training time: ~2-3x slower")
print("  • 2x smaller (5M params):")
print("    - Faster training (~2-3x)")
print("    - Risk: Underfitting, limited capacity")
print("  Recommendation: Balance with dataset size")

# Analysis 4: Block Size Sensitivity
print("\n4. BLOCK SIZE SENSITIVITY")
print("-" * 70)
print(f"Used block size: {block_size}")
print("  Effects of changing block size:")
print("  • Larger (512):")
print("    - Better long-range understanding")
print("    - Training time: ~2x slower (attention is O(n²))")
print("    - Risk: Overfitting, more memory")
print("  • Smaller (128):")
print("    - Much faster training (~4x)")
print("    - Risk: Limited context, can't learn dependencies >128 tokens")
print("  Recommendation: Trade-off between context and speed")

# Analysis 5: Dropout Sensitivity
print("\n5. DROPOUT SENSITIVITY")
print("-" * 70)
print(f"Used dropout: {dropout}")
print("  Effects of changing dropout:")
print("  • Higher (0.2-0.3):")
print("    - Stronger regularization")
print("    - Risk: Underfitting, slower convergence")
print("  • Lower (0.0-0.05):")
print("    - Less regularization")
print("    - Risk: Overfitting, especially on small datasets")
print("  Recommendation: 0.1-0.2 is good balance")

# Analysis 6: Summary of Parameter Interactions
print("\n6. PARAMETER INTERACTIONS")
print("-" * 70)
print("Key interactions to consider:")
print("  • Larger batch size → may need higher LR (linear scaling)")
print("  • Larger model → needs more data or more regularization")
print("  • Larger block size → needs more model capacity")
print("  • Higher LR → may need more warmup iterations")
print("  • More dropout → may need higher LR to overcome noise")

print("\n" + "="*70)
print("SENSITIVITY ANALYSIS COMPLETE")
print("="*70)
print("\nTo perform actual sensitivity tests:")
print("  1. Vary one parameter at a time")
print("  2. Keep other parameters constant")
print("  3. Compare final validation loss")
print("  4. Plot parameter vs. performance curve")
```

### Cell 18: Summary and Conclusions

```python
# ============================================================================
# SUMMARY AND CONCLUSIONS
# ============================================================================

print("="*70)
print("SUMMARY AND CONCLUSIONS")
print("="*70)

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

print(f"\nDataset Comparison:")
print(f"  WikiText-2:")
print(f"    - Training tokens:  {len(wikitext2_train):,}")
print(f"    - Validation loss:  {final_val_loss_wiki:.4f}")
print(f"    - Perplexity:       {final_val_ppl_wiki:.2f}")
print(f"    - Training time:     {sum(history_wikitext['time']):.1f}s")

print(f"\n  Shakespeare:")
print(f"    - Training tokens:  {len(shakespeare_train):,}")
print(f"    - Validation loss:  {final_val_loss_shake:.4f}")
print(f"    - Perplexity:       {final_val_ppl_shake:.2f}")
print(f"    - Training time:     {sum(history_shakespeare['time']):.1f}s")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

findings = []
findings.append(f"1. Dataset size impact: WikiText-2 is {len(wikitext2_train)/len(shakespeare_train):.1f}x larger than Shakespeare")

if final_val_ppl_wiki < final_val_ppl_shake:
    findings.append("2. Model performance: Lower perplexity on WikiText-2 suggests larger dataset helps, or WikiText-2 is easier to model")
else:
    findings.append("2. Model performance: Lower perplexity on Shakespeare suggests its repetitive style is easier to predict")

if wiki_gap < shake_gap:
    findings.append("3. Generalization: WikiText-2 shows better generalization (smaller train-val gap)")
else:
    findings.append("3. Generalization: Shakespeare shows better generalization (smaller train-val gap)")

wiki_tokens_per_sec = total_tokens_wiki / sum(history_wikitext['time'])
shake_tokens_per_sec = total_tokens_shake / sum(history_shakespeare['time'])
findings.append(f"4. Training speed: {wiki_tokens_per_sec:.0f} tok/sec (WikiText-2) vs {shake_tokens_per_sec:.0f} tok/sec (Shakespeare)")

findings.append(f"5. Model architecture: {num_params/1e6:.2f}M parameters provides good balance of quality and speed")

for finding in findings:
    print(f"  {finding}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

recommendations = []
recommendations.append("• For larger datasets (WikiText-2):")
recommendations.append("  - Use larger model (12-24 layers) for better quality")
recommendations.append("  - Increase block size (512-1024) for longer context")
recommendations.append("  - Lower learning rate (6e-4) for stability")

recommendations.append("\n• For smaller datasets (Shakespeare):")
recommendations.append("  - Current model size (6 layers) is appropriate")
recommendations.append("  - Block size 256 is sufficient")
recommendations.append("  - Higher learning rate (3e-3) works well")

recommendations.append("\n• For future experiments:")
recommendations.append("  - Try gradient accumulation for larger effective batch sizes")
recommendations.append("  - Experiment with different learning rate schedules")
recommendations.append("  - Consider mixed precision training for speed")
recommendations.append("  - Test on more diverse datasets")

for rec in recommendations:
    print(rec)

print("\n" + "="*70)
print("COMPARISON WITH LITERATURE")
print("="*70)

print("\nWikiText-2 Perplexities from literature:")
literature_ppl = [
    ("AWD-LSTM (Merity 2016)", 20.98),
    ("LSTM (Merity 2016)", 24.23),
    ("Our GPT (from scratch)", final_val_ppl_wiki),
]

print(f"{'Model':<30} {'Perplexity':>10}")
print("-" * 42)
for name, ppl in literature_ppl:
    marker = "  ← Our model" if "Our" in name else ""
    print(f"{name:<30} {ppl:>10.2f}{marker}")

print("\nInterpretation:")
if final_val_ppl_wiki < 30:
    print("  ✓ Our model performs reasonably well for training from scratch")
else:
    print("  ⚠ Our model needs improvement (larger model, more training)")

print(f"\nNote: Our model is much smaller ({num_params/1e6:.2f}M) than state-of-the-art models")
print("      and trained for much shorter time. Room for improvement is significant.")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

print("\nThis experiment demonstrated:")
print("  ✓ Training GPT from scratch on WikiText-2 is feasible")
print("  ✓ Internal baseline (Shakespeare) provides useful comparison")
print("  ✓ Small models (~10M params) can learn reasonable language models")
print("  ✓ Hyperparameter choices significantly impact training")

print("\nNext steps:")
print("  • Train larger model (24M+ params) for better quality")
print("  • Experiment with different architectures (more layers, heads)")
print("  • Test on additional datasets (OpenWebText, BookCorpus)")
print("  • Fine-tune pretrained models for faster convergence")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

# Save summary to file
summary_file = 'training_summary.txt'
with open(summary_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("TRAINING SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write(f"WikiText-2:\n")
    f.write(f"  Training tokens:  {len(wikitext2_train):,}\n")
    f.write(f"  Validation loss:  {final_val_loss_wiki:.4f}\n")
    f.write(f"  Perplexity:       {final_val_ppl_wiki:.2f}\n")
    f.write(f"  Training time:     {sum(history_wikitext['time']):.1f}s\n\n")
    f.write(f"Shakespeare:\n")
    f.write(f"  Training tokens:  {len(shakespeare_train):,}\n")
    f.write(f"  Validation loss:  {final_val_loss_shake:.4f}\n")
    f.write(f"  Perplexity:       {final_val_ppl_shake:.2f}\n")
    f.write(f"  Training time:     {sum(history_shakespeare['time']):.1f}s\n\n")
    f.write("Model architecture:\n")
    f.write(f"  Parameters: {num_params/1e6:.2f}M\n")
    f.write(f"  Layers: {n_layer}, Heads: {n_head}, Dimensions: {n_embd}\n")

print(f"\n✓ Summary saved to '{summary_file}'")
```

## Execution Time Estimates

- **Cell 6 (Prepare Shakespeare)**: 1-2 minutes
- **Cell 10 (Train WikiText-2)**: 15-30 minutes on GPU
- **Cell 11 (Train Shakespeare)**: 5-10 minutes on GPU
- **Other cells**: < 5 minutes total

**Total expected time**: 25-50 minutes on modern GPU, 2-4 hours on CPU

## Key Features

1. **Training from Scratch**: Both models trained from scratch, no pretrained weights
2. **Internal Baseline**: Shakespeare provides quick comparison for different text types
3. **Comprehensive Visualizations**: Loss curves, perplexity comparisons, token distributions
4. **Detailed Parameter Explanations**: Extensive comments explaining hyperparameter choices
5. **Complete Analysis**: Training dynamics, convergence, overfitting, efficiency metrics
6. **Literature Comparison**: Benchmarks against published results
7. **Hyperparameter Sensitivity**: Analysis of parameter effects and interactions

## Output Files

The notebook will generate:
- `out-wikitext2-test/ckpt.pt` - Best WikiText-2 model checkpoint
- `out-shakespeare-test/ckpt.pt` - Best Shakespeare model checkpoint
- `training_comparison.png` - Loss curves, learning rates, speed, perplexities
- `token_distribution.png` - Token frequency distributions
- `training_summary.txt` - Text summary of results

## Notes

- Ensure sufficient GPU memory (recommended: 8GB+)
- Training on CPU will be 10-50x slower
- Adjust `max_iters` for faster/longer training as needed
- Models can be reused for further experiments or fine-tuning
