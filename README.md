# Retrieval Demo — Sparse Search, Honest About Its Limit

A tiny search engine built from scratch in pure Python: no libraries, no
API keys, no downloads. It indexes 12 short Web3-security documents with
TF-IDF vectors and ranks them for a query with cosine similarity.

## Run it

```
python3 retrieval_demo.py
```

## What it shows

- A demo query ("fake Telegram support asking me to verify my wallet")
  returns the right document first, with scores.
- An evaluation on 5 test queries: **5/5 top-1**.
- A known failure, shown rather than hidden: the query "someone stole my
  recovery words" returns nothing. The right document is the one about
  seed phrases — a seed phrase IS a set of recovery words — but the query
  shares no words with it, so sparse retrieval has nothing to rank.

## Why this matters for Parallel

Parallel builds retrieval for AI agents. This demo shows I understand the
retrieval pipeline end to end: index, vectors, ranking, evaluation — and
its central problem. Sparse vectors can only match shared words. Dense
embedding models exist to close exactly that gap: vectors that capture
meaning, not just vocabulary.

## What this is not, plainly

No neural network was trained here and no embedding model was used. This
is a sparse baseline, and it says so. The next step in my track is dense
embeddings and neural ranking — the math I am working through now, one
piece at a time, starting from vectors.

## Files

- `retrieval_demo.py` — the whole demo, one file, standard library only.
- `README.md` — this file.
