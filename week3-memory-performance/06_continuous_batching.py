"""
WEEK 3, STEP 6: vLLM & continuous batching -- how a production server
handles hundreds of DIFFERENT users' requests at once efficiently,
instead of one request at a time.

GPUs are efficient at parallel work -- processing 8 sequences' next-token
step together costs barely more than processing 1. So real servers batch
multiple users' requests together. The problem: different requests need
wildly different numbers of tokens generated (one user asks a yes/no
question, another asks for an essay).

STATIC BATCHING (the naive approach): group N requests into a batch, run
steps until EVERY sequence in that batch is done, THEN start the next
batch. A short request that finishes in 5 steps but got batched with a
100-step request just sits there occupying a GPU slot doing NOTHING for
95 steps, and -- worse -- no NEW request can start in that freed slot
until the whole batch finishes. Pure waste, and it gets worse the more
request lengths vary.

CONTINUOUS BATCHING (what vLLM, TGI, and similar serving engines
actually do): the instant any sequence in the batch finishes, immediately
replace it with the next waiting request. The GPU's batch slots are never
idle as long as there's ANY queued work, regardless of how uneven request
lengths are. (vLLM specifically also pairs this with PagedAttention --
managing KV cache memory in fixed-size, non-contiguous "pages" instead of
one big reserved block per sequence, so a slot's cache memory can be
freed/reused instantly too, not just its compute slot. This script
simulates the SCHEDULING half of that story, not the memory-paging half
-- that's the same memory-management spirit as step 1's KV cache math,
applied per-slot instead of per-conversation.)

This script runs both scheduling strategies as an actual discrete-step
simulation over the same batch of requests and measures real wasted vs.
useful GPU-slot-steps for each.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import deque
import numpy as np

rng = np.random.default_rng(2)

# a deliberately uneven mix -- some short requests, some long ones -- because
# that's exactly the real-world condition that makes static batching hurt
NUM_REQUESTS = 16
lengths = [int(x) for x in rng.choice([4, 6, 8, 50, 70, 90], size=NUM_REQUESTS)]
BATCH_SIZE = 4

print("=" * 78)
print("SETUP")
print("=" * 78)
print(f"{NUM_REQUESTS} incoming requests, each needing this many tokens generated:")
print(f"  {lengths}")
print(f"GPU can process {BATCH_SIZE} sequences per step, in parallel, at ~equal cost")
print("to processing just 1 (that parallelism is WHY batching is worth doing at all).")
print()


# ---------------------------------- static batching ----------------------------------
def simulate_static_batching(lengths, batch_size):
    total_steps = 0
    useful_slot_steps = 0
    wasted_slot_steps = 0
    batch_log = []
    for i in range(0, len(lengths), batch_size):
        batch = lengths[i:i + batch_size]
        batch_max = max(batch)
        total_steps += batch_max
        useful_slot_steps += sum(batch)
        wasted_slot_steps += sum(batch_max - L for L in batch)
        batch_log.append((batch, batch_max))
    return total_steps, useful_slot_steps, wasted_slot_steps, batch_log


static_steps, static_useful, static_wasted, batch_log = simulate_static_batching(lengths, BATCH_SIZE)

print("=" * 78)
print("STATIC BATCHING -- wait for the WHOLE batch before starting the next one")
print("=" * 78)
for batch, batch_max in batch_log:
    waste_per_seq = [batch_max - L for L in batch]
    print(f"  batch {batch}: runs for {batch_max} steps (the LONGEST member) -- "
          f"idle padding per seq: {waste_per_seq}")
print()
print(f"Total steps to finish everything : {static_steps}")
print(f"Useful (real work) slot-steps     : {static_useful}")
print(f"Wasted (idle-but-blocking) slot-steps: {static_wasted}")
print(f"GPU slot utilization: {static_useful / (static_useful + static_wasted):.1%}")
print()


# ---------------------------------- continuous batching ----------------------------------
def simulate_continuous_batching(lengths, batch_size):
    queue = deque(lengths)
    active = []
    while len(active) < batch_size and queue:
        active.append(queue.popleft())

    total_steps = 0
    useful_slot_steps = 0
    capacity_slot_steps = 0

    while active:
        total_steps += 1
        useful_slot_steps += len(active)          # every occupied slot does REAL work, every step
        capacity_slot_steps += batch_size          # GPU offers this many slots regardless of demand

        remaining = []
        for L in active:
            L -= 1
            if L > 0:
                remaining.append(L)
        active = remaining

        while len(active) < batch_size and queue:  # <-- the whole idea: refill IMMEDIATELY
            active.append(queue.popleft())

    return total_steps, useful_slot_steps, capacity_slot_steps


cont_steps, cont_useful, cont_capacity = simulate_continuous_batching(lengths, BATCH_SIZE)
cont_idle = cont_capacity - cont_useful

print("=" * 78)
print("CONTINUOUS BATCHING -- refill a finished slot with the next request immediately")
print("=" * 78)
print(f"Total steps to finish everything      : {cont_steps}")
print(f"Useful (real work) slot-steps          : {cont_useful}")
print(f"Idle slot-steps (only when queue is truly empty, not from scheduling waste): {cont_idle}")
print(f"GPU slot utilization: {cont_useful / cont_capacity:.1%}")
print()


# ---------------------------------- the comparison ----------------------------------
print("=" * 78)
print("HEAD TO HEAD")
print("=" * 78)
print(f"{'':<28} {'static batching':>18} {'continuous batching':>22}")
print(f"{'total steps to finish all':<28} {static_steps:>18} {cont_steps:>22}")
print(f"{'GPU utilization':<28} {static_useful/(static_useful+static_wasted):>17.1%} "
      f"{cont_useful/cont_capacity:>21.1%}")
print()
print(f"Continuous batching finishes the SAME {NUM_REQUESTS} requests in "
      f"{static_steps - cont_steps} fewer steps ({(static_steps-cont_steps)/static_steps:.1%} faster),")
print("using the exact same GPU, same batch size, same requests -- purely by never")
print("letting a finished sequence's slot sit idle-but-occupied. This is why every")
print("modern LLM serving engine (vLLM, TGI, TensorRT-LLM) is built around continuous")
print("batching as the default -- static batching leaves real throughput on the table")
print("any time request lengths vary, which in production they always do.")
