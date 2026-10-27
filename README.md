# Identity-Bearing Swarm Federation — Reference Implementation

Reference implementation and evaluation harness for the paper
*Identity-Bearing Swarm Federation: Operator-Anchored Cryptographic
Governance for Heterogeneous Multi-Robot Fleets*.

## Archive and source versions

The version 1.0.0 code and data archive is available at
[Zenodo, DOI: 10.5281/zenodo.23551524](https://doi.org/10.5281/zenodo.23551524).
Use this fixed archive when referring to the archived release. The GitHub
`main` branch may contain later documentation updates.

The 27 October 2026 documentation update changes explanatory text, comments
and docstrings without changing executable statements, trial data or saved
reports. The files in `data/` and `reports/` remain the archived evaluation
outputs; they have not been regenerated for this update.

`reports/summary.json` retains the source-size measurements recorded with the
original evaluation: 4,602 statement lines and 7,525 total physical lines.
The current checkout contains 4,663 statement lines (1,611 core, 2,493 harness,
559 tests) and 7,368 total physical lines, measured using `source_size()` in
`scripts/make_tables.py`. The statement-count difference already existed
before this documentation update, following additions to the portable runner
and its tests. The documentation cleanup changes physical line counts only.
Regenerating reports measures the checked-out source version; record its Git
commit when comparing those counts with the frozen evaluation reports.

## What is here

| Path | Contents |
|---|---|
| `ibsf/manifest.py` | Identity manifest (Definition 1) and caste certificates |
| `ibsf/bls.py` | N-party BLS aggregation with operator countersignature (Algorithm 1); participants resolved from the operator-signed roster by bitmap; proof-of-possession helpers. BLS12-381 min-pk via the blst-backed `blspy` binding |
| `ibsf/operator.py` | Operator process: proof-of-possession-checked roster signing, caste issuance, epoch close (every member chain admitted before anything is signed), hash-chained epoch roots, and an `(epoch, seq)` replay guard |
| `ibsf/audit_chain.py` | Per-agent hash-chained audit logs with per-event BLS signatures + fleet Merkle aggregation over identity-bound leaves (each leaf commits to epoch, slot, member key and chain head) |
| `ibsf/certificate.py` | Compliance-certificate export (policy-deduplicated rosters) + complete regulator offline verification (epoch-root signatures, the root-to-root chain that makes an omitted epoch detectable, roster signatures, and their binding) |
| `ibsf/federation.py` | Fleet construction with enforced PoP at commissioning, and end-to-end decision rounds |
| `ibsf/network.py` | Log-normal transport model (median 0.4 ms, p99 4.5 ms) |
| `ibsf/attacks.py` | Five adversarial scenarios (single-robot impersonation at both the manifest and decision surfaces, rogue-key cancellation, caste laundering, whole-tuple operator-signature replay, malformed participant bitmaps) plus the operator-key-compromise bound |
| `ibsf/pbft.py` | PBFT baseline with a quorum-order-statistic round model over the same network substrate |
| `scripts/` | One runner per experiment + `make_tables.py` (single table-generation entry point) + `verify_cert.py` (offline verifier) |
| `data/` | Raw JSONL trial records (latency, attack, reconstruction, failure-injection, and wireless runs are seed-swept, 30 seeds per configuration; certificate-size and PoP-cost runs are single deterministic sweeps) plus the exported certificate under `data/verify_cert/` |
| `scripts/bls_platform_probe.py` | Standalone primitive-cost probe. Imports nothing from this repo and needs no numpy, so it can be copied to any machine with Python and `blspy`. Drop its JSON into `data/` as `platform_probe_<label>.json` and `make_tables.py` picks it up |
| `reports/` | LaTeX table fragments, `summary.json`, `sign_check.json` |

## Methodology note

Cryptographic operations (BLS sign, aggregate, verify; SHA-256
Merkle construction; HMAC authenticators in the PBFT baseline) are
executed for real and measured wall-clock. Network transport is
simulated by the declared log-normal model. The per-decision
decomposition is recorded in every JSONL line, so either component
can be re-derived independently. The PBFT baseline models each
all-to-all round as a quorum order statistic (the round completes
when a quorum of `2f+1` replicas each hold `2f+1` matching
messages), not as the slowest of all links.

Three points where an experiment could otherwise be true by
construction are handled explicitly.

* **Rogue-key scenario.** The adversary runs the textbook cancellation
  attack: it picks its own secret `x` and publishes
  `pk' = g^x * (prod_{i in S} pk_i)^{-1}` over a set `S` of honest
  roster keys, so the aggregate key over `S u {pk'}` collapses to
  `g^x` and it can forge alone. Each trial records whether that attack
  actually succeeds against a naive aggregator
  (`naive_aggregate_admits`) as well as whether our construction
  rejects it, and which of the two defences did so — the substituted
  roster passes the member check and is stopped by the key-list digest
  bound into the operator countersignature.
* **Operator-key compromise.** The adversary is given `sk_O` and
  nothing else. It commissions a roster of keys it generated itself
  and drives a decision under that roster; `honest_keys_used` is
  recorded per trial and is zero, so the 100% admission rate is the
  real bound rather than an artefact of lending the adversary the
  fleet's secrets.
* **Reconstruction.** Each robot is corrupted independently with
  probability `f`, so the number affected is Binomial(N, f) rather than
  a fixed `ceil(f*N)`. With a fixed count the recovered fraction is
  exactly `1-f` in every trial and the confidence interval collapses to
  a point. A report invariant requires nonzero variance for the sampled
  corruption settings. Corruption is
  destructive and takes one of three modes, drawn uniformly:
  `total_loss` (chain erased), `tail_truncation` (random non-empty
  suffix erased), `interior_tamper` (one event's payload mutated in
  place).

  The auditor receives only the signed epoch root, the per-slot head
  commitments, the roster public keys taken from the operator-signed
  roster, and whatever chain data survives. It is not told which slots
  were corrupted, and it is not given the Merkle leaves: it rebuilds
  each one as `epoch_leaf(epoch, slot, member_pk, head)` and
  authenticates the whole vector against the signed root, so the leaf
  vector it works from is both authenticated and attributed to named
  members. It then attempts a proof for every roster slot and every
  event index and verifies per-event BLS signatures.

  Two things come out of that loop, and the second is the claim.
  Because the epoch leaf commits to the chain HEAD, a truncated or
  tampered chain can no longer be replayed to the committed head, so
  none of its events stay provable even though almost all of its bytes
  survive and — for truncation — every surviving signature is still
  valid. `events_present` and `events_signature_verified` are emitted
  next to `events_provable` so the size of that gap is visible.
  Corruption is all-or-nothing per chain under head-only commitment;
  that is a limitation of the construction, measured rather than
  engineered around. What must hold at 100% is DETECTION: every
  corrupted chain is flagged with a concrete reason (absent, broken
  hash link, invalid signature, or head mismatch), no corrupted chain
  contributes a provable event (`silent_acceptances`), and no intact
  chain is wrongly flagged (`intact_chains_flagged`). Detection is
  scored against the runner's ground-truth record of what it corrupted,
  so it can come out below 1. A per-configuration negative control
  additionally mutates one *uncorrupted* chain and confirms the counts
  drop by exactly that chain's events.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
python scripts/run_all.py                 # whole suite in one pass, then reports/
```

By default, `run_all.py` skips the manuscript-number gate: this standalone
artifact does not require a manuscript file. To also
check a local TeX manuscript against the regenerated reports, supply its path:

```bash
python scripts/run_all.py --manuscript /path/to/manuscript.tex
```

The path is checked before experiments start; it must be an existing `.tex`
file. The manuscript check runs last and reports failures normally. The default
still runs all experiments, report aggregation, and certificate verification
(including the report invariants checked by `make_tables.py`). This command
regenerates data; use a separate working copy to preserve the shipped traces.

`run_all.py` is the supported path. It runs every experiment
back-to-back and regenerates `reports/` from the traces it just wrote.
Both properties matter. The reported times are wall-clock measurements
of real BLS and HMAC operations, so taking them on separate occasions
makes tables describing the same construction disagree by tens of
percent depending on machine load; one pass puts them all under one
machine state, and `reports/run_provenance.json` records the load
average the run was taken under. Regenerating only part of `data/` and
leaving `reports/` behind would likewise ship a summary describing
traces that are no longer present.

The individual runners remain available if only one experiment is
wanted, but `make_tables.py` must be re-run after any of them:

```bash
python scripts/run_attacks.py
python scripts/run_sim.py
python scripts/run_pbft_baseline.py
python scripts/run_reconstruction.py
python scripts/run_certsize.py
python scripts/run_epoch_sensitivity.py
python scripts/run_failure_injection.py   # t-of-N liveness + PoP cost
python scripts/run_wireless.py            # Wi-Fi transport sensitivity
python scripts/run_microbench.py          # per-operation micro-costs
python scripts/bls_platform_probe.py --label "my-hardware" \
    > data/platform_probe_my-hardware.json   # optional: your platform
python scripts/make_tables.py             # regenerates all of reports/ from data/
python scripts/verify_cert.py             # verify the COMMITTED artifact + tamper tests
python scripts/verify_cert.py --export    # regenerate data/verify_cert/ first, then verify
```

The seed-swept runners use 30 seeds per configuration, fixed in
each script (the certificate-size and PoP-cost runners are single
deterministic sweeps). `make_tables.py` regenerates every reported table and
`reports/summary.json` from the raw JSONL. Confidence intervals are
cluster bootstraps over seeds, not over individual trials: trials
within a seed share a fleet, a key set and an RNG stream, so the seed
is the independent unit of replication.

`reports/sign_check.json` records the SHA-256 over all trial data, a
per-file digest and record count, the library and platform versions,
and a set of falsifiable invariants recomputed from the traces
(defended scenarios never admitted; the rogue-key cancellation is
exact and does defeat a naive aggregator; the key-compromise bound
borrows no honest key; liveness outcomes track the quorum rule; the
latency decomposition sums to the reported total; reconstruction roots
and leaf vectors verify, every corrupted chain is detected with none
silently accepted and no intact chain falsely flagged, all three
corruption modes actually occur, the recovered fraction has non-zero
variance rather than being `1-f` restated, and the tamper control
fires). `sign_check_pass` is the conjunction of those checks, not a
constant.

`verify_cert.py` verifies the compliance certificate and per-event
inclusion proof already committed under `data/verify_cert/`, using only
those files and the operator public key. It regenerates nothing unless
`--export` is passed, which is the point: a harness that re-exports and
then verifies what it just wrote tests the exporter against itself and
would pass on stale or hand-edited committed files. Use `--export`
after changing a signed format. The event proof is
self-contained rather than a bare Merkle path: it carries the event
body, the signer's public key, the event's BLS signature, the rest of
that agent's chain up to its head, the slot index, and the path from
the identity-bound leaf to the signed fleet root. Verification
independently checks the event signature under the signer key, replays
the hash pointers to the chain head, recomputes the leaf as
`epoch_leaf(epoch, slot, signer_pk, head)`, verifies the Merkle path,
verifies the operator signature over the epoch root carrying that fleet
root, and requires the signer to hold exactly that slot in the epoch's
operator-signed roster.

The exported certificate spans several epochs, because with one epoch
the root chain has nothing to link and an omitted epoch could not be
demonstrated. Ten tamper tests must all fail verification: a flipped
byte in a signed fleet root, an inserted roster with mismatched
membership, an omitted interior epoch, an omitted leading epoch, a
flipped event payload, a swapped signer key (another genuine roster
member), a wrong slot index, a relocated event whose leaf was
recomputed to match the new slot, a truncated chain segment, and a
stripped event signature. In the epoch-omission cases every surviving
root still carries a valid operator signature; only the root-to-root
chain shows the gap.

What the root chain establishes is that the exported window starts at
the operator's first epoch and has no interior gaps. It does not, on
its own, establish that the window runs to the present: dropping a
suffix of the most recent epochs leaves an internally consistent chain,
and catching that needs a freshness anchor outside the certificate.

## Cite

For the archived code and data, cite
[Identity-Bearing Swarm Federation, version 1.0.0](https://doi.org/10.5281/zenodo.23551524).
For changes on GitHub, also record the exact commit used. A paper citation
will be added when a published reference is available.

## License

MIT — see `LICENSE`.
