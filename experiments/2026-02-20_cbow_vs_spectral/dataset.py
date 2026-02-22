"""Dataset utilities for word embedding experiments.

Supports:
- Text8 (quick experiments, ~17M words)
- FineWeb-Edu (same corpus as embeddings-js project)
"""

import os
import re
import glob
import urllib.request
import zipfile
import collections
import numpy as np
import torch
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


# ── Text8 ────────────────────────────────────────────────────────

def download_text8():
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, 'text8')
    if os.path.exists(filepath):
        return filepath
    zip_path = os.path.join(DATA_DIR, 'text8.zip')
    print("Downloading Text8...")
    urllib.request.urlretrieve('http://mattmahoney.net/dc/text8.zip', zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    return filepath


def load_text8(max_words=None):
    filepath = download_text8()
    with open(filepath) as f:
        words = f.read().split()
    if max_words:
        words = words[:max_words]
    print(f"Text8: {len(words):,} words")
    return words


# ── FineWeb-Edu (Parquet) ────────────────────────────────────────

def tokenize(text):
    """Lowercase, keep alphanumeric, split on whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if len(w) >= 2]


def load_fineweb(max_words=None, num_shards=1):
    """Load FineWeb-Edu from Parquet files.

    Searches for .parquet files in:
      1. ../embeddings-js/data/parquet/  (shared with JS project)
      2. ./data/parquet/                 (local copy)
    """
    js_dir = os.path.normpath(
        os.path.join(BASE_DIR, '..', '..', '..', 'embeddings-js', 'data', 'parquet')
    )
    local_dir = os.path.join(DATA_DIR, 'parquet')

    parquet_dir = None
    for d in [js_dir, local_dir]:
        if os.path.isdir(d) and glob.glob(os.path.join(d, '*.parquet')):
            parquet_dir = d
            break

    if not parquet_dir:
        raise FileNotFoundError(
            "No FineWeb-Edu Parquet files found.\n"
            "Options:\n"
            "  1. Download via embeddings-js:  cd ../embeddings-js && npm run download\n"
            "  2. Place .parquet files in ./data/parquet/\n"
            "  3. Use --dataset text8 for quick testing"
        )

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pip install pyarrow  (needed to read Parquet files)")

    files = sorted(glob.glob(os.path.join(parquet_dir, '*.parquet')))[:num_shards]
    print(f"Loading {len(files)} Parquet shard(s) from {parquet_dir}")

    words = []
    for fpath in files:
        table = pq.read_table(fpath, columns=['text'])
        for text in table.column('text').to_pylist():
            words.extend(tokenize(text))
            if max_words and len(words) >= max_words:
                break
        if max_words and len(words) >= max_words:
            break

    words = words[:max_words] if max_words else words
    print(f"FineWeb-Edu: {len(words):,} words")
    return words


# ── Corpus loader (unified) ──────────────────────────────────────

def load_corpus(source='text8', max_words=None, **kwargs):
    if source == 'text8':
        return load_text8(max_words)
    elif source == 'fineweb':
        return load_fineweb(max_words, **kwargs)
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'text8' or 'fineweb'.")


# ── Vocabulary & preprocessing ───────────────────────────────────

def build_vocab(words, min_count=5, max_vocab=30000):
    counter = collections.Counter(words)
    pairs = [(w, c) for w, c in counter.items() if c >= min_count]
    pairs.sort(key=lambda x: -x[1])
    pairs = pairs[:max_vocab]

    word2idx = {}
    idx2word = []
    counts = np.zeros(len(pairs), dtype=np.float64)

    for i, (word, count) in enumerate(pairs):
        word2idx[word] = i
        idx2word.append(word)
        counts[i] = count

    print(f"Vocabulary: {len(word2idx):,} words (min_count={min_count})")
    return word2idx, idx2word, counts


def prepare_data(words, word2idx, counts, subsample_t=1e-5):
    """Convert words to indices with subsampling of frequent words."""
    total = counts.sum()
    freqs = counts / total
    # P(keep) = sqrt(t/f) + t/f  (Mikolov 2013)
    keep_prob = np.minimum(np.sqrt(subsample_t / freqs) + subsample_t / freqs, 1.0)

    indices = []
    for w in words:
        if w in word2idx:
            idx = word2idx[w]
            if np.random.random() < keep_prob[idx]:
                indices.append(idx)

    data = np.array(indices, dtype=np.int64)
    print(f"Training tokens: {len(data):,} (subsampled from {len(words):,})")
    return data


def noise_distribution(counts, power=0.75):
    """Noise distribution for negative sampling: freq^0.75."""
    dist = counts ** power
    dist /= dist.sum()
    return torch.from_numpy(dist).float()


# ── PyTorch Dataset ──────────────────────────────────────────────

class CBOWDataset(Dataset):
    """Returns (context_indices, target_index) for CBOW training."""

    def __init__(self, data, window_size=5):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - 2 * self.window_size

    def __getitem__(self, idx):
        center = idx + self.window_size
        target = self.data[center]
        context = np.concatenate([
            self.data[center - self.window_size:center],
            self.data[center + 1:center + 1 + self.window_size]
        ])
        return torch.from_numpy(context.copy()), torch.tensor(target, dtype=torch.long)
