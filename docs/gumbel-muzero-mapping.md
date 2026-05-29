# Gumbel-MuZero in lc0: mctx → lc0 mapping

Reference: Danihelka et al., **"Policy improvement by planning with Gumbel"**,
ICLR 2022 (https://openreview.net/forum?id=bERaNdoegnO).
Authoritative implementation: DeepMind's **mctx** library
(https://github.com/google-deepmind/mctx).

This document maps every mctx component we need to its target location in
lc0's classic search.  All formulas + constants are taken verbatim from
mctx as of commit-of-record at fetch time.  No paraphrasing — when in
doubt, read this doc against the mctx source.

---

## 0. High-level summary

Gumbel-MuZero replaces THREE things in AlphaZero-style MCTS:

1. **Root action selection** — PUCT → "Sequential Halving with Gumbel"
2. **Non-root action selection** — PUCT → "Full Gumbel MuZero deterministic
   selection" (which is NOT PUCT)
3. **Training target** — visit-count distribution → softmax of
   `prior_logits + completed_qvalues` (no Gumbel noise in training target!)

The Gumbel noise is sampled ONCE per search at the root.  It influences
ONLY root action selection and the final played-move choice.  Non-root
search and training target are deterministic given the search outcome.

---

## 1. The Gumbel noise

**mctx (`policies.py`, `gumbel_muzero_policy`)**:
```python
rng_key, gumbel_rng = jax.random.split(rng_key)
gumbel = gumbel_scale * jax.random.gumbel(
    gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)
```

Sampled once per search.  `gumbel_scale` default = 1.0.  Standard
Gumbel(0, 1) per action.  Gumbel sample formula: `-log(-log(U))` where
`U ~ Uniform(0,1)`.

**lc0 target**: stored as `std::vector<float> root_gumbel_` on `Search`,
sized to number of legal root moves.  Filled once at search start.

---

## 2. Sequential halving schedule

**mctx (`seq_halving.py`, `get_sequence_of_considered_visits`)**:

Builds a pre-computed schedule mapping each simulation index → "considered
visit count" (= the visit count at which an action is eligible for this
simulation).

```python
def get_sequence_of_considered_visits(max_num_considered_actions, num_simulations):
  if max_num_considered_actions <= 1:
    return tuple(range(num_simulations))
  log2max = int(math.ceil(math.log2(max_num_considered_actions)))
  sequence = []
  visits = [0] * max_num_considered_actions
  num_considered = max_num_considered_actions
  while len(sequence) < num_simulations:
    num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
    for _ in range(num_extra_visits):
      sequence.extend(visits[:num_considered])
      for i in range(num_considered):
        visits[i] += 1
    num_considered = max(2, num_considered // 2)
  return tuple(sequence[:num_simulations])
```

For `m=16, N=250`: log2max=4, sequence length=250, has shape something
like `[0,0,...,0 (16 times),1,1,...,1 (16 times), 0,0,...,0 (8 times),
1,1,...,1 (8 times), 2,2,...,2 (8 times), ...]`.  Pre-computable;
deterministic given (m, N).

**lc0 target**: `std::vector<int> considered_visit_schedule_` computed at
search start (memoized per (m, N) pair if needed).

---

## 3. Score function (root)

**mctx (`seq_halving.py`, `score_considered`)**:

```python
def score_considered(considered_visit, gumbel, logits, normalized_qvalues,
                     visit_counts):
  low_logit = -1e9
  logits = logits - jnp.max(logits, keepdims=True, axis=-1)
  penalty = jnp.where(visit_counts == considered_visit, 0, -jnp.inf)
  return jnp.maximum(low_logit, gumbel + logits + normalized_qvalues) + penalty
```

Three steps:
1. Normalize logits: `logits' = logits - max(logits)` (so max becomes 0).
2. Penalty filter: action is eligible iff `visit_counts == considered_visit`.
3. Score among eligible: `max(-1e9, gumbel + logits' + normalized_q)`.

The "normalized_qvalues" here is the σ-transformed-completed-Q vector
(see §4).

**lc0 target**: free function in `search.cc`'s anonymous namespace,
`ScoreConsidered(considered_visit, gumbel[i], logits[i], normalized_q[i],
visit_counts[i])` returning a float per action.  Caller does argmax.

---

## 4. Q-completion + σ transform

**mctx (`qtransforms.py`, `qtransform_completed_by_mix_value`)**:

```python
def qtransform_completed_by_mix_value(
    tree, node_index, *,
    value_scale: float = 0.1,
    maxvisit_init: float = 50.0,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: float = 1e-8) -> Array:
  # Read raw Q-values and visit counts at node
  ...
  # Compute mixed value for unvisited actions
  raw_value = tree.raw_values[node_index]  # value-head prediction at this node
  mixed_value = _compute_mixed_value(raw_value, qvalues, visit_counts, prior_probs)
  # Replace unvisited Q with mixed_value
  completed = _complete_qvalues(qvalues, visit_counts=visit_counts, value=mixed_value)
  # Rescale to [0, 1]
  if rescale_values:
    completed = _rescale_qvalues(completed, epsilon)
  # σ transform
  maxvisit = jnp.max(visit_counts, axis=-1)
  visit_scale = maxvisit_init + maxvisit
  return visit_scale * value_scale * completed
```

Helper formulas:

```python
def _compute_mixed_value(raw_value, qvalues, visit_counts, prior_probs):
  sum_visit_counts = sum(visit_counts)
  prior_probs = max(eps_small, prior_probs)
  sum_probs = sum(prior_probs * (visit_counts > 0))
  weighted_q = sum(prior_probs * qvalues * (visit_counts > 0) / sum_probs)
  return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1)

def _complete_qvalues(qvalues, visit_counts, value):
  return where(visit_counts > 0, qvalues, value)

def _rescale_qvalues(q, epsilon):
  min_v = min(q)
  max_v = max(q)
  return (q - min_v) / max(max_v - min_v, epsilon)
```

**Defaults**: maxvisit_init = 50.0, value_scale = 0.1, rescale_values = True,
use_mixed_value = True, epsilon = 1e-8.

**lc0 target**: helper functions in `search.cc` namespace:
- `ComputeMixedValue(raw_value, qvalues, visit_counts, prior_probs)` → float
- `CompleteQValues(qvalues, visit_counts, mixed)` → `vector<float>`
- `RescaleQValues(qvalues, epsilon)` → `vector<float>` (rescales to [0,1])
- `SigmaQTransform(rescaled_q, max_visit, c_visit, c_scale)` → `vector<float>`
- Wrapper `QTransformCompletedByMixValue(node, params)` → `vector<float>`

`raw_value` for lc0 = the value-head prediction at this node from the
network output.  Available via `node->GetWL()` or similar (need to verify
the right accessor).

---

## 5. Root action selection (during search)

**mctx (`action_selection.py`, `gumbel_muzero_root_action_selection`)**:

```python
def gumbel_muzero_root_action_selection(rng_key, tree, node_index, *,
                                          num_simulations,
                                          max_num_considered_actions,
                                          qtransform):
  visit_counts = tree.children_visits[node_index]
  prior_logits = tree.children_prior_logits[node_index]
  completed_qvalues = qtransform(tree, node_index)  # σ(q) — already transformed
  table = get_table_of_considered_visits(max_num_considered_actions, num_simulations)
  num_valid_actions = sum(1 - tree.root_invalid_actions)
  num_considered = min(max_num_considered_actions, num_valid_actions)
  simulation_index = sum(visit_counts)
  considered_visit = table[num_considered, simulation_index]
  gumbel = tree.extra_data.root_gumbel
  to_argmax = score_considered(considered_visit, gumbel, prior_logits,
                                completed_qvalues, visit_counts)
  return masked_argmax(to_argmax, tree.root_invalid_actions)
```

Key invariants:
- `simulation_index = sum of all visit counts so far at root`.  Used as
  the index into the precomputed schedule.
- `considered_visit = schedule[num_considered][simulation_index]`.  Tells
  which visit count an action must currently have to be eligible.
- Within eligible actions, pick argmax of `gumbel + logits + σ(q)`.

**lc0 target**: replace the root selection in `PickNodesToExtendTask` for
the `is_root_node` branch when `use_gumbel_muzero=true`.  Per-batch logic:
- For each visit to dispatch:
  - Compute `simulation_index = node->GetN() + n_in_flight` (sum of visits
    + in-flight)
  - Look up `considered_visit = schedule[m][simulation_index]`
  - Among root edges with `nstarted == considered_visit`, pick argmax of
    `gumbel + logits + σ(q)`
  - Dispatch one visit there (IncrementNInFlight)
- This is per-visit dispatch, NOT per-batch dispatch.  Need to think
  about how to batch this with lc0's picker (mctx does single-visit;
  lc0 wants batches of visits per backend eval).

---

## 6. Non-root (interior) action selection

**mctx (`action_selection.py`, `gumbel_muzero_interior_action_selection`)**:

```python
def gumbel_muzero_interior_action_selection(rng_key, tree, node_index, depth, *,
                                              qtransform):
  visit_counts = tree.children_visits[node_index]
  prior_logits = tree.children_prior_logits[node_index]
  completed_qvalues = qtransform(tree, node_index)  # σ(q)
  to_argmax = _prepare_argmax_input(
      probs=softmax(prior_logits + completed_qvalues),
      visit_counts=visit_counts)
  return argmax(to_argmax)

def _prepare_argmax_input(probs, visit_counts):
  to_argmax = probs - visit_counts / (1 + sum(visit_counts))
  return to_argmax
```

Concretely: at every non-root node, the selected action is

```
argmax_a [ π_a − N_a / (1 + sum_N) ]
where π_a = softmax(logits + σ(q))_a
```

This is a "completed-search-improved-policy minus visit-frequency"
formulation.  No Gumbel noise, no PUCT exploration constant.

**lc0 target**: replace PUCT in the non-root branch of
`PickNodesToExtendTask`'s inner loop, when `use_gumbel_muzero=true`.
The score becomes `π_a − N_a / (1 + sum_N)` per edge; pick argmax.

This is structurally simpler than PUCT (no cpuct hyperparameter, no
sqrt, no FPU) but requires computing the completed-Q + softmax per
inner-loop iteration.

---

## 7. Final action selection (played move)

**mctx (`policies.py`, end of `gumbel_muzero_policy`)**:

After search completes:
```python
considered_visit = jnp.max(summary.visit_counts, axis=-1)
completed_qvalues = qtransform(search_tree, ROOT_INDEX)
to_argmax = score_considered(considered_visit, gumbel, root.prior_logits,
                              completed_qvalues, summary.visit_counts)
action = masked_argmax(to_argmax, invalid_actions)
```

The played action is "best from among actions with max visit count, by
gumbel-perturbed completed-policy logits."

**lc0 target**: replace `GetBestRootChildWithTemperature` (or rather,
gate it on `use_gumbel_muzero`).  When Gumbel-MuZero is on, compute the
final action via this formula instead of temperature sampling over
visit counts.

Side note: this means **no temperature mechanism is needed with
Gumbel-MuZero** — the stochasticity comes from the Gumbel noise.
`--temperature` settings are ignored when `use_gumbel_muzero=true`.

---

## 8. Training target (policy)

**mctx (`policies.py`, end of `gumbel_muzero_policy`)**:

```python
completed_search_logits = _mask_invalid_actions(
    root.prior_logits + completed_qvalues, invalid_actions)
action_weights = jax.nn.softmax(completed_search_logits)
```

**Key**: the training target is `softmax(prior_logits + σ(q))`.  **No
Gumbel noise.**  Just the network's policy logits adjusted by the
σ-transformed completed Q values from the search tree.

**lc0 target**: replace `GetTrainingTargetVisits` (or add a parallel
path) with `GetGumbelMuZeroPolicyTarget` returning the softmax of
`(prior_logit[i] + σ(q)[i])` for each root edge, normalized to
probability distribution.  Written into `processed_visits` (existing
wiring).  Reuses V7's `probabilities[1858]` field.

---

## 9. lc0-side decisions / open questions

### A. Per-visit vs per-batch dispatch at root

mctx's algorithm is per-visit: each simulation index picks one action.
lc0's picker dispatches multiple visits per batch (for backend efficiency).

**Plan**: at root, dispatch one visit at a time (per the algorithm), then
fan out into the descended subtree using interior selection (which can
batch normally because interior selection is deterministic and doesn't
depend on the visit being completed first — it just uses current
visit_counts which are stable within a batch).

This means root-level becomes more sequential than current lc0
(roughly one root-edge pick per batch dispatch, not several).  Might
hurt throughput slightly.  Acceptable for v1.

### B. Q estimates within a batch

Interior selection computes `softmax(logits + σ(q)) − N/(1+sumN)`.  This
depends on completed-Q which uses current Q estimates.  Within a single
backend batch (visits dispatched but not yet completed), Q is stale.

**Plan**: use current Q (whatever it is at batch-start) for all visits
in the batch.  Same as lc0's current PUCT behavior — it uses cached Q
within a picker batch.  No behavior change relative to AZ paradigm here.

### C. Tree reuse

Confirmed off for selfplay (`game.cc:72`, `kReuseTreeId = false`).  Each
search starts fresh tree.  Fresh Gumbel noise each search.  No tree
reuse complications.

### D. Mutual exclusivity with forced exploration, PTP, advisor

When `use_gumbel_muzero=true`:
- `forced_exploration_factor` ignored (Gumbel SH does that role)
- `use_policy_target_pruning` ignored (improved-policy target is the
  pruning mechanism)
- `advisor_min_visits` ?? — open question, see below
- Temperature ignored (Gumbel noise provides stochasticity)
- FPU ignored at non-root (mctx doesn't have FPU; Q-completion uses
  mixed_value instead)

### E. Advisor integration with Gumbel-MuZero

Options:
1. **Force-include**: advisor's move always in initial m-action set
   (m-1 by Gumbel + 1 advisor).  Get same per-round visit treatment.
2. **Ignore**: don't combine.  Run pure Gumbel-MuZero, no advisor signal.
3. **Boost logits**: add a constant bonus to advisor's logit before
   Gumbel sampling, biasing the top-m selection.

**Plan**: implement option 1 (force-include) as the default behavior
when both `use_gumbel_muzero=true` and an advisor is configured.

### F. Batched/multi-threaded picker

lc0 has task workers operating on child subtrees concurrently.  Each task
worker has its own workspace.  Interior selection at task-worker nodes
needs the same completed-Q computation.  This means computing σ(q)
inside the task worker's inner loop.

Cost: σ(q) requires reading children's Q, computing mixed_value (which
iterates over visited children), rescaling, then scaling.  This is more
expensive than PUCT's straightforward arithmetic.

Mitigation: cache σ(q) per node, invalidate only when a child's N or Q
changes.  Within an inner-loop iteration that picks the same node
repeatedly, no recomputation.

### G. Numerical stability

- Q rescaling: when all Q values are equal (min==max), epsilon=1e-8
  prevents divide-by-zero.  Result: all rescaled Q values are 0 (since
  numerator is 0).  Sane behavior.
- mixed_value: when sum_visit_counts is 0, falls back to `raw_value`.
- softmax in training target: standard numerical-stable softmax (subtract
  max first).
- Gumbel sampling: -log(-log(U)) where U is Uniform(0,1) avoiding 0
  exactly to prevent log(0).  Use `std::uniform_real_distribution<float>`
  with `nextafter(0, 1)` as minimum.

---

## 10. Phase plan (revised based on mctx reading)

**Phase 1.1** ✓ DONE — option scaffolding (commit `dc77ab9`).  Defaults
to be updated post-mctx-read: `c_scale 1.0 → 0.1`.

**Phase 1.2** — Q transform + Gumbel sampling utilities.
- `qtransform_completed_by_mix_value` C++ port
- Gumbel sample utility (one-liner from std::uniform_real_distribution)
- Sequential halving schedule precompute
- All tested independently against mctx outputs on toy inputs.

**Phase 1.3** — Root action selection.
- New `SequentialHalvingState` on Search
- Override root selection in `PickNodesToExtendTask`
- Per-visit dispatch loop (one root pick per iteration, then descend)

**Phase 1.4** — Non-root action selection (Full Gumbel deterministic).
- Replace PUCT at non-root nodes when `use_gumbel_muzero=true`.
- Score formula: `softmax(logits + σ(q)) − N/(1+sumN)`.
- Bigger change than root — touches every level of the tree.

**Phase 1.5** — Final action selection.
- Replace `GetBestRootChildWithTemperature` (or gate it) for Gumbel mode.

**Phase 1.6** — Training target.
- New `GetGumbelMuZeroPolicyTarget` returning the constructed improved
  policy.
- Wire into `Search::GetTrainingTargetVisits` callsite (game.cc).
- Set V7 `reserved[0] = 1.0f` to flag the chunk as Gumbel-MuZero.

**Phase 1.7** — Advisor force-include.

**Phase 1.8** — Validation against mctx.
- Toy positions, side-by-side run, byte-exact comparison of:
  - σ(q) values
  - Root selection decisions
  - Final improved-policy distribution

**Phase 2** — A/B training run vs AZ baseline.

---

## 11. Reference URLs

- Paper: https://openreview.net/forum?id=bERaNdoegnO
- mctx repo: https://github.com/google-deepmind/mctx
- Key files:
  - `mctx/_src/policies.py` — top-level orchestrator
  - `mctx/_src/action_selection.py` — root + interior selection
  - `mctx/_src/qtransforms.py` — Q completion + σ transform
  - `mctx/_src/seq_halving.py` — sequential halving schedule + score
  - `mctx/_src/search.py` — search loop
