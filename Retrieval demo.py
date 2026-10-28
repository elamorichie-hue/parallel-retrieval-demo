"""Retrieval demo: a tiny search engine built from scratch.

Pure Python, no libraries, no API keys, no downloads.
Method: TF-IDF vectors + cosine similarity over 12 short Web3-security
documents. It ranks documents for a query, evaluates itself on 5 test
queries, and then shows its own honest failure: a query whose words do
not overlap with the right document. Sparse vectors cannot see meaning
beyond shared words. That failure is exactly the problem dense
embedding models exist to solve.
"""

import math
import re
from collections import Counter

DOCS = [
    ("Fake support DM asking for wallet verification",
     "A fake Telegram support account sends you a direct message asking you "
     "to connect your wallet to verify your account. Real support never "
     "messages first and never asks you to connect a wallet to verify anything."),
    ("Flash USDT that disappears",
     "Flash USDT is fake Tether that shows up in your wallet balance and then "
     "disappears when you try to send it. Scammers use it to fake a payment "
     "or a giveaway win."),
    ("Phishing link posing as a wallet update",
     "A phishing link pretends to be a wallet app update. It opens a fake "
     "login page that steals your password and seed phrase when you type them in."),
    ("Giveaway form stealing seed phrases",
     "A fake giveaway asks you to enter your seed phrase into a form to claim "
     "a prize. Anyone with your seed phrase owns your wallet."),
    ("Admin impersonator charging an unlock fee",
     "An admin impersonator in a group chat offers to unlock your funds if you "
     "send a small fee first. The fee is the theft; there is nothing to unlock."),
    ("Airdrop that demands a gas fee",
     "A fake airdrop promises free tokens but requires a gas fee payment "
     "before delivery. Real airdrops never ask you to pay to receive tokens."),
    ("Malicious approval draining a wallet",
     "A malicious contract approval lets a scammer drain your wallet later. "
     "Always check what an approval allows before you sign it."),
    ("Legitimate security announcement",
     "A project posts a security announcement in its official channel. It "
     "warns about scams, shares no links, and asks you for nothing."),
    ("Hardware wallet phishing email",
     "A phishing email pretends to come from a hardware wallet company and "
     "warns that your device will be locked unless you download an update."),
    ("Fake support website live chat",
     "A fake support website runs a live chat that asks for your credentials "
     "and wallet details to fix a problem that does not exist."),
    ("Guaranteed returns trading group",
     "A trading group promises guaranteed returns and shows screenshots of "
     "big wins. Guaranteed returns are a promise no honest trader can make."),
    ("How to verify a contract address",
     "Before approving a contract, verify its address on a block explorer and "
     "compare it with the project's official announcement."),
]

STOP = {"a", "an", "the", "to", "of", "in", "and", "for", "with",
        "your", "you", "it", "is", "are", "be", "on", "from", "that",
        "this", "me", "my", "if", "by", "as", "at", "or", "do", "does"}


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def build_index(docs):
    tokenized = [tokens(title + " " + text) for title, text in docs]
    n = len(docs)
    df = Counter()
    for toks in tokenized:
        for term in set(toks):
            df[term] += 1
    idf = {term: math.log(n / count) for term, count in df.items()}
    vectors, norms = [], []
    for toks in tokenized:
        vec = {}
        for term, count in Counter(toks).items():
            vec[term] = (1 + math.log(count)) * idf[term]
        vectors.append(vec)
        norms.append(math.sqrt(sum(v * v for v in vec.values())))
    return idf, vectors, norms


IDF, VECTORS, NORMS = build_index(DOCS)


def search(query, top=3):
    qvec = {}
    for term, count in Counter(tokens(query)).items():
        if term in IDF:
            qvec[term] = (1 + math.log(count)) * IDF[term]
    qnorm = math.sqrt(sum(v * v for v in qvec.values()))
    scored = []
    for i, vec in enumerate(VECTORS):
        dot = sum(qvec[t] * vec.get(t, 0.0) for t in qvec)
        score = dot / (qnorm * NORMS[i]) if qnorm and NORMS[i] else 0.0
        scored.append((score, i))
    scored.sort(reverse=True)
    return [(score, DOCS[i][0]) for score, i in scored[:top]]


def show(query, top=3):
    print(f'Query: "{query}"')
    results = search(query, top)
    if not results or results[0][0] <= 0:
        print("  No matching documents. The query shares no words with any "
              "document, so sparse retrieval has nothing to rank.")
        return None
    for rank, (score, title) in enumerate(results, 1):
        print(f"  {rank}. [{score:.3f}] {title}")
    return results[0][1]


def main():
    print("=== Demo query ===")
    show("fake Telegram support asking me to verify my wallet")
    print()
    print("=== Evaluation: 5 word-matching queries ===")
    tests = [
        ("fake support account asking to verify wallet",
         "Fake support DM asking for wallet verification"),
        ("USDT disappears from my wallet",
         "Flash USDT that disappears"),
        ("phishing link wallet update",
         "Phishing link posing as a wallet update"),
        ("giveaway asking for seed phrase",
         "Giveaway form stealing seed phrases"),
        ("admin fee to unlock funds",
         "Admin impersonator charging an unlock fee"),
    ]
    hits = 0
    for query, expected in tests:
        top = show(query, top=1)
        ok = top == expected
        hits += ok
        print(f"  Expected: {expected} -> {'HIT' if ok else 'MISS'}")
        print()
    print(f"Word-matching queries: {hits}/{len(tests)} top-1.")
    print()
    print("=== Known failure: vocabulary mismatch ===")
    print("The right document is 'Giveaway form stealing seed phrases':")
    print("a seed phrase IS a set of recovery words. But the query shares")
    print("no words with that document, so sparse retrieval fails here.")
    show("someone stole my recovery words")
    print()
    print("That miss is the whole case for dense embeddings: vectors that")
    print("capture meaning, not just shared words.")


if __name__ == "__main__":
    main()
