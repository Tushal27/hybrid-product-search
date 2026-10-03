"""
WEEK 5, STEP 7: Agent memory -- how an agent recalls something from way
earlier in a conversation (or a past session entirely) that has long
since fallen out of its context window.

week3-memory-performance/01_context_window.py proved the context window
is a hard limit, and that the simplest fix -- a sliding window keeping
only the most recent N tokens -- means anything older is GONE, with zero
regard for whether it was actually important. Agent memory is the real
answer to that: instead of keeping recent-in-time facts, keep
RELEVANT-to-right-now facts, retrieved by MEANING, not recency.

THREE KINDS, ALL DEMONSTRATED HERE:
  SHORT-TERM : whatever's currently in the context window -- free, but
               bounded and recency-biased (this IS week 3's sliding window).
  LONG-TERM  : facts persisted somewhere outside the context window
               entirely (here: a plain Python list -- a real system uses a
               database) so they survive even past the current conversation.
  VECTOR     : long-term memories retrieved by SEMANTIC similarity to the
               current query -- reusing week1-embeddings' exact cosine-
               similarity technique on real GloVe vectors, not new machinery.

The proof: store several unrelated facts, ask a question that shares NO
exact keywords with the one relevant fact, and show vector retrieval
finds it anyway by MEANING, then feeds it to the model so it actually
uses it in its answer.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import gensim.downloader as api
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

print("=" * 78)
print("SETUP -- reuse week1-embeddings' GloVe vectors for real semantic similarity")
print("=" * 78)
wv = api.load("glove-wiki-gigaword-50")


def embed(text):
    """Mean-pooled GloVe embedding -- a simple, real sentence embedding: average
    the vector of every known word. (A proper sentence-embedding model would do
    better, but this is the exact same cosine-similarity mechanism at work.)"""
    words = [w.strip(".,!?").lower() for w in text.split()]
    vectors = [wv[w] for w in words if w in wv]
    if not vectors:
        return np.zeros(wv.vector_size)
    return np.mean(vectors, axis=0)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


# ---------------------------------- long-term memory store ----------------------------------
MEMORIES = [
    "The user's favorite color is blue.",
    "The user has a severe peanut allergy.",
    "The user's dog is named Rex.",
    "The user works as a night-shift nurse.",
    "The user's favorite food is pizza.",
    "The user is learning to play the guitar.",
    "The user's birthday is in March.",
    "The user recently moved to a new apartment.",
]
memory_embeddings = [embed(m) for m in MEMORIES]

print(f"Long-term memory store: {len(MEMORIES)} facts, persisted OUTSIDE any context window")
for m in MEMORIES:
    print(f"  - {m}")
print()


def retrieve_top_k(query, k=2):
    query_emb = embed(query)
    scored = [(cosine_similarity(query_emb, mem_emb), mem) for mem_emb, mem in zip(memory_embeddings, MEMORIES)]
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


# ---------------------------------- 1. prove retrieval works by MEANING, not keywords ----------------------------------
print("=" * 78)
print("1. RETRIEVAL BY MEANING -- zero shared keywords with the relevant fact")
print("=" * 78)
QUERY = "Does this person have any food restrictions?"
print(f"Query: {QUERY!r}")
shared_words = set(w.lower().strip(".,") for w in QUERY.split()) & \
               set(w.lower().strip(".,") for w in "The user has a severe peanut allergy.".split())
print(f"Words this query shares with the peanut-allergy memory: {shared_words or '(none)'}")
print()

results = retrieve_top_k(QUERY, k=3)
print("Top-3 memories by cosine similarity:")
for score, mem in results:
    print(f"  {score:.3f}  {mem}")
print()
top_memory = results[0][1]
print(f"Highest-scoring memory: {top_memory!r}")
print(f"Found the SAFETY-CRITICAL fact despite zero keyword overlap: "
      f"{'peanut' in top_memory}")
print()

print("-" * 78)
print("HONEST COUNTEREXAMPLE -- this technique is NOT magic, and query phrasing matters")
print("-" * 78)
HARD_QUERY = "What should I be careful about when cooking dinner for this person?"
hard_results = retrieve_top_k(HARD_QUERY, k=3)
print(f"Query: {HARD_QUERY!r}")
for score, mem in hard_results:
    print(f"  {score:.3f}  {mem}")
found_it = "peanut" in hard_results[0][1]
print(f"Found the allergy fact at #1 this time: {found_it}")
print("Mean-pooled GloVe (average every word's vector, no training for THIS task) is")
print("a genuinely weak sentence embedding -- it's sensitive to phrasing in a way a")
print("dedicated sentence-embedding model (e.g. sentence-transformers, or an")
print("embeddings API) would not be nearly as much. This is a real, honest limit of")
print("the SPECIFIC embedding technique used here, not of vector memory as an idea --")
print("production RAG/memory systems invest specifically in embedding model quality")
print("because retrieval is only as good as the vectors it's built on.")
print()


# ---------------------------------- 2. feed retrieved memory into an actual answer ----------------------------------
print("=" * 78)
print("2. USE THE RETRIEVED MEMORY -- feed it to the model as context")
print("=" * 78)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
model.eval()


def ask(messages, max_new_tokens=60):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


retrieved_context = "\n".join(f"- {m}" for _, m in results[:2])
messages = [
    {"role": "system", "content": f"Relevant facts about the user, retrieved from long-term memory:\n"
                                   f"{retrieved_context}\n\nUse these facts if relevant to answer the user."},
    {"role": "user", "content": QUERY},
]
answer = ask(messages)
print(f"Retrieved context injected into the prompt:\n{retrieved_context}")
print(f"\nModel's answer: {answer.strip()!r}")
print(f"Mentions the allergy: {'allerg' in answer.lower() or 'peanut' in answer.lower()}")
print()

# ---------------------------------- 3. why this beats a pure sliding window ----------------------------------
print("=" * 78)
print("3. WHY THIS BEATS WEEK 3'S SLIDING WINDOW FOR OLD-BUT-IMPORTANT FACTS")
print("=" * 78)
print(f"If the peanut-allergy fact were mentioned once, {len(MEMORIES)*20} conversation turns ago,")
print("and the context window only fit the most recent handful of turns (week 3,")
print("step 1), a sliding window would have discarded it completely by now -- no")
print("way to recover it, no matter how relevant it becomes later. Vector memory")
print("doesn't care WHEN a fact was stored, only whether it's semantically relevant")
print("to what's being asked RIGHT NOW -- retrieval cost stays roughly constant")
print("whether there are 8 memories or 8 million, because it's a similarity search,")
print("not 'keep scrolling back through everything that was ever said.'")
