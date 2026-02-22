"""Train CBOW word embeddings and evaluate.

Usage:
    python train_cbow.py --dataset text8 --epochs 5
    python train_cbow.py --dataset text8 --max-words 1000000 --epochs 2   # quick test
    python train_cbow.py --dataset fineweb --epochs 5                      # real experiment
"""

import argparse
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_corpus, build_vocab, prepare_data, noise_distribution, CBOWDataset
from cbow import CBOW
from evaluate import run_evaluation

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def train(args):
    print(f"\n{'=' * 60}")
    print(f"CBOW Training")
    print(f"{'=' * 60}")
    print(f"Dataset:     {args.dataset}")
    print(f"Dim:         {args.dim}")
    print(f"Window:      {args.window}")
    print(f"Negatives:   {args.negatives}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"LR:          {args.lr}")
    print(f"{'=' * 60}\n")

    # ── Data ──────────────────────────────────────────────────────
    words = load_corpus(args.dataset, max_words=args.max_words)
    word2idx, idx2word, counts = build_vocab(
        words, min_count=args.min_count, max_vocab=args.max_vocab
    )
    data = prepare_data(words, word2idx, counts)
    noise = noise_distribution(counts)

    dataset = CBOWDataset(data, window_size=args.window)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # ── Model ─────────────────────────────────────────────────────
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Device: {device}")
    print(f"Samples: {len(dataset):,}")
    print(f"Batches/epoch: {len(loader):,}\n")

    model = CBOW(len(word2idx), args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for i, (context, target) in enumerate(loader):
            context = context.to(device)
            target = target.to(device)

            # Sample negatives on CPU, move to device
            neg = torch.multinomial(
                noise, args.batch_size * args.negatives, replacement=True
            ).view(args.batch_size, args.negatives).to(device)

            loss = model(context, target, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if i > 0 and i % 2000 == 0:
                elapsed = time.time() - t0
                wps = i * args.batch_size / elapsed
                print(f"  [{epoch+1}/{args.epochs}] batch {i:,}/{len(loader):,}  "
                      f"loss={total_loss/n_batches:.4f}  {wps:,.0f} w/s")

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={total_loss/n_batches:.4f}  ({elapsed:.1f}s)")

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, 'cbow_model.pt')
    torch.save({
        'model': model.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word,
        'args': vars(args),
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # ── Evaluate ──────────────────────────────────────────────────
    embeddings = model.get_embeddings()
    results = run_evaluation(embeddings, word2idx, idx2word)

    results_path = os.path.join(SAVE_DIR, 'cbow_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train CBOW on Text8 or FineWeb-Edu')
    p.add_argument('--dataset', default='text8', choices=['text8', 'fineweb'])
    p.add_argument('--dim', type=int, default=300, help='Embedding dimension')
    p.add_argument('--window', type=int, default=5, help='Context window size')
    p.add_argument('--negatives', type=int, default=5, help='Negative samples per example')
    p.add_argument('--epochs', type=int, default=5, help='Training epochs')
    p.add_argument('--batch-size', type=int, default=512, help='Batch size')
    p.add_argument('--lr', type=float, default=0.001, help='Learning rate (Adam)')
    p.add_argument('--min-count', type=int, default=5, help='Min word frequency')
    p.add_argument('--max-vocab', type=int, default=30000, help='Max vocabulary size')
    p.add_argument('--max-words', type=int, default=None, help='Limit corpus size (for testing)')
    args = p.parse_args()
    train(args)
