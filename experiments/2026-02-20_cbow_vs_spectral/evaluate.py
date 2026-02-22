"""Evaluation benchmarks for word embeddings.

WordSim-353, SimLex-999, Google Analogy, nearest neighbors.
Benchmark datasets are downloaded automatically on first use.
"""

import os
import urllib.request
import zipfile
import numpy as np
from scipy.stats import spearmanr

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'eval')


def _ensure_dir():
    os.makedirs(EVAL_DIR, exist_ok=True)


def _download(url, filepath):
    if os.path.exists(filepath):
        return True
    try:
        print(f"  Downloading {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


# ── Benchmark loaders ────────────────────────────────────────────

def load_wordsim353():
    _ensure_dir()
    filepath = os.path.join(EVAL_DIR, 'wordsim353.tab')

    if not os.path.exists(filepath):
        zip_path = os.path.join(EVAL_DIR, 'ws353.zip')
        url = 'http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip'
        if _download(url, zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(EVAL_DIR)
            os.remove(zip_path)
            # The zip may contain combined.csv or combined.tab
            for name in ['combined.csv', 'combined.tab']:
                src = os.path.join(EVAL_DIR, name)
                if os.path.exists(src):
                    os.rename(src, filepath)
                    break

    pairs = []
    if not os.path.exists(filepath):
        return pairs
    with open(filepath) as f:
        for line in f:
            sep = '\t' if '\t' in line else ','
            parts = line.strip().split(sep)
            if len(parts) >= 3:
                try:
                    pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
                except ValueError:
                    continue  # header
    return pairs


def load_simlex999():
    _ensure_dir()
    filepath = os.path.join(EVAL_DIR, 'SimLex-999.txt')

    if not os.path.exists(filepath):
        zip_path = os.path.join(EVAL_DIR, 'simlex.zip')
        url = 'https://fh295.github.io/SimLex-999.zip'
        if _download(url, zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(EVAL_DIR)
            os.remove(zip_path)
            src = os.path.join(EVAL_DIR, 'SimLex-999', 'SimLex-999.txt')
            if os.path.exists(src):
                os.rename(src, filepath)

    pairs = []
    if not os.path.exists(filepath):
        return pairs
    with open(filepath) as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                try:
                    pairs.append((parts[0].lower(), parts[1].lower(), float(parts[3])))
                except (ValueError, IndexError):
                    continue
    return pairs


def load_google_analogy():
    _ensure_dir()
    filepath = os.path.join(EVAL_DIR, 'questions-words.txt')
    url = 'https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt'
    _download(url, filepath)

    questions = []
    categories = {}
    category = None

    if not os.path.exists(filepath):
        return questions, categories
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(':'):
                category = line[2:]
                categories[category] = []
            else:
                parts = line.lower().split()
                if len(parts) == 4:
                    q = tuple(parts)
                    questions.append(q)
                    if category:
                        categories[category].append(q)

    return questions, categories


# ── Evaluation functions ─────────────────────────────────────────

def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-10)


def eval_similarity(embeddings, word2idx, pairs, name=""):
    """Spearman correlation between model cosine similarity and human scores."""
    normed = normalize(embeddings)
    model_scores, human_scores = [], []

    for w1, w2, score in pairs:
        if w1 in word2idx and w2 in word2idx:
            cos = float(np.dot(normed[word2idx[w1]], normed[word2idx[w2]]))
            model_scores.append(cos)
            human_scores.append(score)

    if len(model_scores) < 10:
        print(f"  {name}: too few pairs ({len(model_scores)})")
        return 0.0, len(model_scores)

    rho, p = spearmanr(human_scores, model_scores)
    print(f"  {name}: rho={rho:.4f}  (p={p:.2e}, {len(model_scores)}/{len(pairs)} pairs)")
    return float(rho), len(model_scores)


def eval_analogy(embeddings, word2idx, idx2word, questions, name=""):
    """Analogy accuracy via 3CosAdd: a:b :: c:? -> d = argmax cos(x, b-a+c)."""
    normed = normalize(embeddings)
    correct, total = 0, 0

    for a, b, c, d in questions:
        if not all(w in word2idx for w in [a, b, c, d]):
            continue

        ia, ib, ic, id_ = word2idx[a], word2idx[b], word2idx[c], word2idx[d]
        query = normed[ib] - normed[ia] + normed[ic]
        query /= np.linalg.norm(query) + 1e-10

        sims = normed @ query
        for excl in [ia, ib, ic]:
            sims[excl] = -2

        if np.argmax(sims) == id_:
            correct += 1
        total += 1

    if total == 0:
        print(f"  {name}: no evaluable questions")
        return 0.0, 0

    acc = correct / total
    print(f"  {name}: accuracy={acc:.4f}  ({correct}/{total})")
    return float(acc), total


def eval_nearest_neighbors(embeddings, word2idx, idx2word, words, k=8):
    normed = normalize(embeddings)
    print("\n  Nearest Neighbors:")
    for word in words:
        if word not in word2idx:
            continue
        idx = word2idx[word]
        sims = normed @ normed[idx]
        sims[idx] = -2
        top = np.argsort(sims)[-k:][::-1]
        nbrs = ", ".join(f"{idx2word[i]}({sims[i]:.3f})" for i in top[:5])
        print(f"    {word}: {nbrs}")


# ── Full evaluation suite ────────────────────────────────────────

def run_evaluation(embeddings, word2idx, idx2word):
    """Run all benchmarks. Returns results dict."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    results = {}

    # Similarity
    print("\n--- Word Similarity ---")
    for name, loader in [("WordSim-353", load_wordsim353), ("SimLex-999", load_simlex999)]:
        try:
            pairs = loader()
            if pairs:
                rho, n = eval_similarity(embeddings, word2idx, pairs, name)
                results[name] = {'spearman': rho, 'pairs': n}
            else:
                print(f"  {name}: dataset not available")
        except Exception as e:
            print(f"  {name}: {e}")

    # Analogy
    print("\n--- Word Analogy ---")
    try:
        questions, categories = load_google_analogy()
        if questions:
            acc, n = eval_analogy(embeddings, word2idx, idx2word, questions, "Google (all)")
            results['analogy_all'] = {'accuracy': acc, 'total': n}

            sem = [q for c, qs in categories.items() if not c.startswith('gram') for q in qs]
            syn = [q for c, qs in categories.items() if c.startswith('gram') for q in qs]
            if sem:
                a, n = eval_analogy(embeddings, word2idx, idx2word, sem, "  Semantic")
                results['analogy_semantic'] = {'accuracy': a, 'total': n}
            if syn:
                a, n = eval_analogy(embeddings, word2idx, idx2word, syn, "  Syntactic")
                results['analogy_syntactic'] = {'accuracy': a, 'total': n}
        else:
            print("  Dataset not available")
    except Exception as e:
        print(f"  Analogy: {e}")

    # Nearest neighbors (always works)
    test_words = ['king', 'queen', 'man', 'woman', 'good', 'bad',
                  'dog', 'cat', 'computer', 'france', 'paris']
    eval_nearest_neighbors(embeddings, word2idx, idx2word, test_words)

    return results
