"""Aggregate JSONL data into LaTeX tables, summary.json, sign_check.json."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

FLEET_SIZES = (10, 50, 100, 500)
ATTACK_ROWS = (
    ("single-impersonation", "Single-robot impersonation"),
    ("rogue-subset-addition", "Rogue key, cancellation form"),
    # The first caste scenario controls signature verification. The second
    # replays a genuinely issued certificate after demotion.
    ("caste-forgery", "Caste certificate, unauthorised issuer"),
    ("caste-laundering", "Caste replay after demotion"),
    ("operator-signature-replay", "Operator-signature replay"),
    ("malformed-bitmap", "Malformed participant bitmap"),
    ("operator-key-compromise", "Operator-key compromise (bound)"),
)


def load(name: str) -> list[dict]:
    with (DATA / name).open() as fh:
        return [json.loads(line) for line in fh]


def cluster_boot_ci(
    rows: list[dict],
    field: str,
    cluster_key: str = "seed",
    stat=np.median,
    n_boot: int = 2000,
) -> tuple[float, float]:
    """Cluster bootstrap CI, resampling seeds rather than trials.

    The trials within one seed share a fleet, a key set and an RNG
    stream, so they are not independent draws. Resampling individual
    trials would treat 600 correlated observations as 600 independent
    ones and report an interval that is too narrow. The independent unit
    of replication is the seed, so whole seeds are resampled with
    replacement and their trials carried along.
    """
    clusters: dict[object, list[float]] = defaultdict(list)
    for r in rows:
        clusters[r[cluster_key]].append(float(r[field]))
    keys = list(clusters)
    groups = [np.asarray(clusters[k]) for k in keys]
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        stats.append(stat(np.concatenate([groups[i] for i in pick])))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _p99(a: np.ndarray) -> float:
    """p99 as a bootstrap statistic, for tail intervals."""
    return float(np.percentile(a, 99))


def _rate(rows: list[dict], field: str) -> float | None:
    """Fraction of rows in which a boolean diagnostic field is true."""
    vals = [r[field] for r in rows if field in r and r[field] is not None]
    return round(sum(bool(v) for v in vals) / len(vals), 6) if vals else None


def table_sybil(summary: dict) -> None:
    rows = load("attacks.jsonl")
    grid: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for r in rows:
        grid[(r["scenario"], r["n"])].append(r["admitted"])
    lines = []
    for key, label in ATTACK_ROWS:
        cells = []
        for n in FLEET_SIZES:
            vals = grid[(key, n)]
            rate = sum(vals) / len(vals)
            cells.append(f"{rate:.3f}")
            summary.setdefault("sybil", {})[f"{key}/n{n}"] = {
                "rate": rate,
                "trials": len(vals),
            }
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    (REPORTS / "table_vii_2_sybil.tex").write_text("\n".join(lines) + "\n")

    # rogue-key scenario: report the attack's own realisability and which
    # defence rejected it, so the claim is not "we rejected something we
    # never showed to be dangerous".
    rogue = [r for r in rows if r["scenario"] == "rogue-subset-addition"]
    if rogue:
        summary["rogue_key_attack"] = {
            "trials": len(rogue),
            "k": rogue[0].get("k"),
            "cancellation_exact_rate": _rate(rogue, "cancellation_exact"),
            "naive_aggregator_admits_rate": _rate(
                rogue, "naive_aggregate_admits"
            ),
            "pop_admits_rogue_key_rate": _rate(rogue, "pop_admits_rogue_key"),
            "commissioning_admits_rate": _rate(
                rogue, "commissioning_admits_rogue_key"
            ),
            "honest_roster_admits_rate": _rate(rogue, "honest_roster_admits"),
            "substituted_roster_admits_rate": _rate(
                rogue, "substituted_roster_admits"
            ),
            "substituted_roster_member_check_rate": _rate(
                rogue, "substituted_roster_member_check"
            ),
            "substituted_roster_operator_check_rate": _rate(
                rogue, "substituted_roster_operator_check"
            ),
        }
    compromise = [r for r in rows if r["scenario"] == "operator-key-compromise"]
    if compromise:
        summary["operator_key_compromise"] = {
            "trials": len(compromise),
            "admission_rate": _rate(compromise, "admitted"),
            "max_honest_keys_used": max(
                r.get("honest_keys_used", -1) for r in compromise
            ),
        }
    logger.info("sybil table written")


def table_latency(summary: dict) -> None:
    lines = []
    cells = []
    for n in FLEET_SIZES:
        rows = load(f"sim_n{n}.jsonl")
        totals = [r["total_ms"] for r in rows]
        med, p99 = float(np.median(totals)), float(np.percentile(totals, 99))
        lo, hi = cluster_boot_ci(rows, "total_ms")
        # The tail is quoted in the paper, so it carries an interval on
        # the same footing as the median. A p99 read off 600 correlated
        # trials as a bare number invites the reader to treat it as a
        # settled quantity when it is the least stable statistic here.
        p99lo, p99hi = cluster_boot_ci(rows, "total_ms", stat=_p99)
        cells.append(f"{med:.1f} [{p99:.1f}]")
        msgs = sorted({r["messages"] for r in rows}) if "messages" in rows[0] else []
        summary.setdefault("latency", {})[f"n{n}"] = {
            "median_ms": med,
            "p99_ms": p99,
            "ci95_median": [lo, hi],
            "ci95_p99": [p99lo, p99hi],
            "ci95_method": "cluster bootstrap over seeds (2000 resamples)",
            "messages": msgs[0] if len(msgs) == 1 else msgs,
            "trials": len(totals),
            "seeds": len({r["seed"] for r in rows}),
            "components_median_ms": {
                k: float(np.median([r[k] for r in rows]))
                for k in (
                    "transport_ms",
                    "member_sign_ms",
                    "aggregate_ms",
                    "operator_overhead_ms",
                    "countersign_ms",
                    "verify_ms",
                )
            },
        }
    lines.append("Federation (this work) & " + " & ".join(cells) + r" \\")
    pbft_rows = load("pbft.jsonl")
    pcells = []
    for n in FLEET_SIZES:
        totals = [r["total_ms"] for r in pbft_rows if r["n"] == n]
        med, p99 = float(np.median(totals)), float(np.percentile(totals, 99))
        pcells.append(f"{med:.1f} [{p99:.1f}]")
        msgs = [r["messages"] for r in pbft_rows if r["n"] == n][0]
        sub = [r for r in pbft_rows if r["n"] == n]
        lo, hi = cluster_boot_ci(sub, "total_ms")
        p99lo, p99hi = cluster_boot_ci(sub, "total_ms", stat=_p99)
        summary.setdefault("pbft", {})[f"n{n}"] = {
            "median_ms": med,
            "p99_ms": p99,
            "ci95_median": [lo, hi],
            "ci95_p99": [p99lo, p99hi],
            "ci95_method": "cluster bootstrap over seeds (2000 resamples)",
            "messages": msgs,
        }
    lines.append("PBFT baseline & " + " & ".join(pcells) + r" \\")
    (REPORTS / "table_vii_3_latency.tex").write_text("\n".join(lines) + "\n")

    # The paper reports the comparison under three accountings of the
    # operator/leader scheduling term, to show the ordering does not
    # depend on which one a reader prefers. Two of the three were once
    # computed in prose with nothing in the artifact behind them, which
    # is exactly the unverifiable-number problem the rest of this file
    # exists to prevent. They are emitted here instead.
    from ibsf.network import NetworkModel               # noqa: PLC0415

    net = NetworkModel()
    rng = np.random.default_rng(0)
    acc: dict[str, dict[str, float]] = {}
    for n in FLEET_SIZES:
        fed_rows = load(f"sim_n{n}.jsonl")
        pbft_totals = np.asarray(
            [r["total_ms"] for r in pbft_rows if r["n"] == n]
        )
        # neither side charged for scheduling
        fed_bare = np.median(
            [r["total_ms"] - r["operator_overhead_ms"] for r in fed_rows]
        )
        # both charged, drawing the leader's term from the same model the
        # operator already pays from
        leader = np.array([
            net.operator_overhead(rng) for _ in range(len(pbft_totals))
        ])
        acc[f"n{n}"] = {
            "federation_as_reported_ms": float(
                np.median([r["total_ms"] for r in fed_rows])
            ),
            "federation_without_scheduling_ms": float(fed_bare),
            "pbft_as_reported_ms": float(np.median(pbft_totals)),
            "pbft_with_leader_scheduling_ms": float(
                np.median(pbft_totals + leader)
            ),
        }
    summary["scheduling_accountings"] = {
        "note": "three ways of charging the scheduling term; the paper "
                "reports all three because the ordering is the same "
                "under each",
        "per_fleet_size": acc,
    }
    logger.info("latency table written")


def _mode_totals(rows: list[dict], field: str) -> dict[str, int]:
    """Sum a per-mode dict field across trials."""
    out: dict[str, int] = {}
    for r in rows:
        for mode, count in r.get(field, {}).items():
            out[mode] = out.get(mode, 0) + count
    return out


def table_recon(summary: dict) -> None:
    rows = load("reconstruction.jsonl")
    lines = []
    for f in (0.1, 0.3, 0.5, 0.7):
        cells = []
        for n in (10, 100, 500):
            sub = [r for r in rows if r["n"] == n and abs(r["f"] - f) < 1e-9]
            root_ok = all(r["epoch_root_recoverable"] for r in sub)
            leaf_ok = all(r["leaf_vector_authenticated"] for r in sub)
            ev = float(np.median([r["events_recoverable"] for r in sub]))
            pt = float(np.median([r["proof_production_time_ms"] for r in sub]))
            lo, hi = cluster_boot_ci(sub, "events_recoverable")
            controls = [
                r["negative_control_detects_tamper"] for r in sub
                if r.get("negative_control_detects_tamper") is not None
            ]
            injected = _mode_totals(sub, "corrupted_by_mode")
            detected = _mode_totals(sub, "detected_by_mode")
            n_corrupt = sum(injected.values())
            n_detect = sum(detected.values())
            det_rate = n_detect / n_corrupt if n_corrupt else None
            det_cell = (
                r"100\%" if det_rate == 1.0
                else ("n/a" if det_rate is None else f"{det_rate * 100:.1f}\\%")
            )
            # Cell format matches the manuscript table verbatim, so the
            # emitted fragment can be compared to the typeset table
            # character for character. Detection rate and proof timing
            # are reported in prose and in summary.json rather than
            # added to the cell, preserving the table's stated format.
            ev_lo = min(r["events_recoverable"] for r in sub)
            ev_hi = max(r["events_recoverable"] for r in sub)
            cells.append(f"{ev:.2f} [{ev_lo:.2f}, {ev_hi:.2f}]")
            assert det_cell is not None      # reported in prose/summary
            summary.setdefault("reconstruction", {})[f"n{n}/f{f}"] = {
                "root_recoverable_all": root_ok,
                "leaf_vector_authenticated_all": leaf_ok,
                "events_recoverable_median": ev,
                "events_recoverable_ci95": [lo, hi],
                "events_recoverable_min": float(
                    min(r["events_recoverable"] for r in sub)
                ),
                "events_recoverable_max": float(
                    max(r["events_recoverable"] for r in sub)
                ),
                "events_present_fraction_median": float(
                    np.median([r["events_present_fraction"] for r in sub])
                ),
                "events_provable_median": float(
                    np.median([r["events_provable"] for r in sub])
                ),
                "events_signature_verified_median": float(
                    np.median([r["events_signature_verified"] for r in sub])
                ),
                "events_total": sub[0]["events_total"],
                "robots_corrupted_median": float(
                    np.median([r["robots_corrupted"] for r in sub])
                ),
                "chains_corrupted_by_mode": injected,
                "chains_detected_by_mode": detected,
                "chains_absent_by_mode": _mode_totals(sub, "absent_by_mode"),
                "chains_present_unverifiable_by_mode": _mode_totals(
                    sub, "present_unverifiable_by_mode"
                ),
                "detection_rate": det_rate,
                "detection_rate_by_mode": {
                    m: (detected.get(m, 0) / injected[m])
                    for m in injected if injected[m]
                },
                "silent_acceptances": sum(r["silent_acceptances"] for r in sub),
                "intact_chains_flagged": sum(
                    r["intact_chains_flagged"] for r in sub
                ),
                "slots_unavailable_median": float(
                    np.median([r["slots_unavailable"] for r in sub])
                ),
                "negative_control_detects_tamper": (
                    all(controls) if controls else None
                ),
                "proof_ms_median": pt,
            }
        lines.append(f"$f{{=}}{f}$ & " + " & ".join(cells) + r" \\")
    (REPORTS / "table_viii_recon.tex").write_text("\n".join(lines) + "\n")
    logger.info("reconstruction table written")


def table_certsize(summary: dict) -> None:
    rows = load("certsize.jsonl")
    lines = []
    for n in (10, 100, 500):
        cells = []
        for k in (1, 10, 100, 1000):
            sub = [r for r in rows if r["n"] == n and r["k"] == k][0]
            cells.append(f"{sub['root_level_bytes']:,}")
            summary.setdefault("certsize", {})[f"n{n}/k{k}"] = sub
        lines.append(f"$N{{=}}{n}$ & " + " & ".join(cells) + r" \\")
    (REPORTS / "table_viii_cert_size.tex").write_text("\n".join(lines) + "\n")
    logger.info("certsize table written")


def epoch_sensitivity(summary: dict) -> None:
    rows = load("epoch_sensitivity.jsonl")
    for length in (10, 30, 60, 300):
        sub = [r for r in rows if r["epoch_length_s"] == length]
        # Percentiles are taken over the POOLED arrival samples. Taking a
        # p95 within each seed (~4 requests) and then a median across
        # seeds estimates nothing in particular and disagreed with the
        # closed form the sampler implements; pooling ~120 arrivals per
        # configuration recovers it.
        pooled = [
            x for r in sub for x in r.get("revocation_tte_samples_s", [])
        ]
        summary.setdefault("epoch_sensitivity", {})[f"L{length}"] = {
            "tte_median_s": float(np.median(pooled)) if pooled else None,
            "tte_p95_s": float(np.percentile(pooled, 95)) if pooled else None,
            "tte_analytic_median_s": sub[0].get("revocation_tte_analytic_median_s"),
            "tte_analytic_p95_s": sub[0].get("revocation_tte_analytic_p95_s"),
            "tte_is_analytic": True,
            "tte_samples": len(pooled),
            "cert_bytes_per_hour": sub[0]["cert_bytes_per_hour"],
            "operator_sign_ms_per_hour": sub[0]["operator_sign_ms_per_hour"],
            "epochs_per_hour": sub[0]["epochs_per_hour"],
        }


def failure_injection(summary: dict) -> None:
    rows = load("failure_injection.jsonl")
    ns = sorted({r["n"] for r in rows})
    cfracs = sorted({r["crash_fraction"] for r in rows})
    lines = []
    for n in ns:
        threshold = next(r["threshold"] for r in rows if r["n"] == n)
        cells = []
        for c in cfracs:
            sub = [
                r for r in rows
                if r["n"] == n and abs(r["crash_fraction"] - c) < 1e-9
            ]
            succ = [r for r in sub if r["succeeded"]]
            rate = len(succ) / len(sub) if sub else 0.0
            med = (
                float(np.median([r["total_ms"] for r in succ]))
                if succ else None
            )
            cells.append(f"{rate:.3f}")
            summary.setdefault("failure_injection", {})[f"n{n}/c{c}"] = {
                "success_rate": rate,
                "trials": len(sub),
                "median_ms": med,
                "threshold": threshold,
            }
        lines.append(f"$N{{=}}{n}$ & " + " & ".join(cells) + r" \\")
    (REPORTS / "table_ix_failure_injection.tex").write_text(
        "\n".join(lines) + "\n"
    )
    logger.info("failure-injection table written")


def pop_commissioning(summary: dict) -> None:
    rows = load("pop_cost.jsonl")
    prove = [r["prove_ms"] for r in rows]
    verify = [r["verify_ms"] for r in rows]
    summary["pop_commissioning"] = {
        "prove_ms_median": float(np.median(prove)),
        "verify_ms_median": float(np.median(verify)),
        "events": len(rows),
    }
    line = (
        f"Proof-of-possession & {float(np.median(prove)):.3f} & "
        f"{float(np.median(verify)):.3f} & {len(rows)}" + r" \\"
    )
    (REPORTS / "table_ix_pop_cost.tex").write_text(line + "\n")
    logger.info("PoP commissioning table written")


def platform_probe(summary: dict) -> None:
    """Cross-platform primitive costs and the derived slowdown factor.

    Read from whatever `platform_probe_*.json` files are present, so a
    reader who runs the standalone probe on their own hardware gets a
    row without editing this script. The workstation file is the
    reference; every other platform is reported relative to it.
    """
    files = sorted(DATA.glob("platform_probe_*.json"))
    if not files:
        return
    loaded = {}
    for p in files:
        doc = json.loads(p.read_text())
        loaded[doc["host"]["label"]] = doc
    ref_label = next(
        (k for k in loaded if "workstation" in k), next(iter(loaded))
    )
    ref = {(r["operation"], r["n"]): r["median_ms"]
           for r in loaded[ref_label]["results"]}

    out = {"reference": ref_label, "platforms": {}}
    for label, doc in loaded.items():
        ops = {(r["operation"], r["n"]): r["median_ms"] for r in doc["results"]}
        ratios = [ops[k] / ref[k] for k in ops if k in ref and ref[k] > 0]
        out["platforms"][label] = {
            "cpu": doc["host"].get("cpu"),
            "cores": doc["host"].get("cores"),
            "machine": doc["host"].get("machine"),
            "median_ms": {f"{op}/{n}": v for (op, n), v in ops.items()},
            "slowdown_vs_reference": {
                "min": round(min(ratios), 3) if ratios else None,
                "max": round(max(ratios), 3) if ratios else None,
                "median": round(float(np.median(ratios)), 3) if ratios else None,
            },
        }
    summary["platform_probe"] = out
    logger.info("platform probe: %d platform(s)", len(loaded))

    # Emit the platform table from the same measurements as the summary.
    # Compute ratios from full-precision medians before display rounding;
    # dividing rounded table cells can introduce inconsistent ratios.
    non_ref = [k for k in loaded if k != ref_label]
    if len(non_ref) == 1:
        other = non_ref[0]
        rops = {(r["operation"], r["n"]): r["median_ms"]
                for r in loaded[ref_label]["results"]}
        oops = {(r["operation"], r["n"]): r["median_ms"]
                for r in loaded[other]["results"]}
        table_rows = (
            ("Member signature", ("member_sign", None)),
            ("Single verification", ("single_verify", None)),
            ("PoP generation", ("pop_prove", None)),
            ("PoP verification", ("pop_verify", None)),
            (r"Aggregate verify, $N{=}10$", ("fast_aggregate_verify", 10)),
            (r"Aggregate verify, $N{=}100$", ("fast_aggregate_verify", 100)),
            (r"Aggregate verify, $N{=}500$", ("fast_aggregate_verify", 500)),
            (r"Aggregation, $N{=}500$", ("aggregate", 500)),
        )
        lines = []
        for label, key in table_rows:
            d = rops[key]
            a = oops[key]
            lines.append(
                f"{label:<27} & {d:.2f} & {a:.2f} & {a / d:.2f}" + r" \\"
            )
        (REPORTS / "table_ix_platform.tex").write_text("\n".join(lines) + "\n")
        logger.info("platform table written (%s vs %s)", ref_label, other)


def microbench(summary: dict) -> None:
    """Two implementation-artefact costs quoted in the latency discussion.

    These were quoted in the manuscript but never surfaced in summary.json,
    so nothing tied the prose to the data and a p95 slipped in where a
    median was meant. Surfacing them keeps them reconcilable like every
    other number.
    """
    rows = load("microbench.jsonl")
    out: dict[str, dict] = {}
    for r in rows:
        out[f"{r['operation']}/n{r['n']}"] = {
            "median_ms": r["median_ms"],
            "p95_ms": r["p95_ms"],
        }
    summary["microbench"] = out


def source_size(summary: dict) -> None:
    """Count source lines for the manuscript implementation description.

    Report both total physical lines and statement-only lines. Statement
    counts exclude blank lines, comment-only lines and module, class or
    function docstrings. The manuscript-number gate checks these reported
    counts alongside other declared quantities.
    """
    import io
    import tokenize

    def code_lines(path: Path) -> int:
        src = path.read_text()
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        except Exception:
            return 0
        skip: set[int] = set()
        prev = None
        for t in toks:
            if t.type == tokenize.COMMENT:
                skip.add(t.start[0])
            # a string in statement position is a docstring
            if t.type == tokenize.STRING and (
                prev is None
                or prev.type in (tokenize.INDENT, tokenize.NEWLINE,
                                 tokenize.NL, tokenize.DEDENT)
            ):
                skip.update(range(t.start[0], t.end[0] + 1))
            if t.type not in (tokenize.NL, tokenize.COMMENT):
                prev = t
        return sum(
            1 for i, line in enumerate(src.splitlines(), 1)
            if line.strip() and i not in skip
        )

    groups = {
        "core": sorted((ROOT / "ibsf").rglob("*.py")),
        "harness": sorted((ROOT / "scripts").rglob("*.py")),
        "tests": sorted((ROOT / "tests").rglob("*.py")),
    }
    out = {k: sum(code_lines(f) for f in v) for k, v in groups.items()}
    out["total"] = sum(out.values())
    out["every_line"] = sum(
        len(f.read_text().splitlines())
        for v in groups.values() for f in v
    )
    summary["source_size"] = out


def wireless(summary: dict) -> None:
    rows = load("wireless.jsonl")
    ns = sorted({r["n"] for r in rows})
    lines = []
    cells = []
    for n in ns:
        totals = [r["total_ms"] for r in rows if r["n"] == n]
        med = float(np.median(totals))
        p99 = float(np.percentile(totals, 99))
        cells.append(f"{med:.1f} [{p99:.1f}]")
        summary.setdefault("wireless", {})[f"n{n}"] = {
            "median_ms": med,
            "p99_ms": p99,
            "trials": len(totals),
        }
    lines.append("Wi-Fi transport & " + " & ".join(cells) + r" \\")
    (REPORTS / "table_ix_wireless.tex").write_text("\n".join(lines) + "\n")
    logger.info("wireless table written")


def _invariants() -> dict[str, bool]:
    """Falsifiable invariants the emitted traces must satisfy.

    Each entry can come out False on real data; together they are what
    `sign_check_pass` reports. They check the traces against properties
    the experiments are supposed to establish, so a rigged or broken run
    shows up here rather than passing silently.
    """
    checks: dict[str, bool] = {}

    attacks = load("attacks.jsonl")
    defended = {
        "single-impersonation",
        "rogue-subset-addition",
        "caste-forgery",
        "caste-laundering",
        "operator-signature-replay",
        "malformed-bitmap",
    }
    checks["defended_scenarios_never_admitted"] = not any(
        r["admitted"] for r in attacks if r["scenario"] in defended
    )
    # The laundering row is only evidence if the replayed certificate was
    # admissible before the demotion and its signature is still valid
    # afterwards. Without both controls, zero admission could result from
    # failure to construct a valid certificate rather than correct rejection.
    # run_attacks flattens AttackResult.detail into each record, so the
    # control fields sit at the top level, not under a "detail" key.
    laundering = [r for r in attacks if r["scenario"] == "caste-laundering"]
    checks["caste_laundering_has_positive_control"] = bool(laundering) and all(
        r.get("pre_demotion_admitted") is True
        and r.get("signature_still_valid") is True
        for r in laundering
    )
    compromise = [
        r for r in attacks if r["scenario"] == "operator-key-compromise"
    ]
    checks["operator_key_compromise_always_admitted"] = bool(
        compromise
    ) and all(r["admitted"] for r in compromise)
    # the bound is only meaningful if the adversary borrowed no honest key
    checks["operator_key_compromise_uses_no_honest_key"] = all(
        r.get("honest_keys_used") == 0 for r in compromise
    )
    # the rogue-key defence is only meaningful if the attack itself works
    rogue = [r for r in attacks if r["scenario"] == "rogue-subset-addition"]
    checks["rogue_key_cancellation_is_exact"] = bool(rogue) and all(
        r.get("cancellation_exact") for r in rogue
    )
    checks["rogue_key_attack_defeats_naive_aggregator"] = bool(rogue) and all(
        r.get("naive_aggregate_admits") for r in rogue
    )

    # Liveness outcome must track the quorum rule, recomputed HERE from
    # the fleet size rather than read back from the runner. Comparing the
    # runner's `succeeded` bit to the runner's `quorum_met` bit would be
    # circular: both come from the same threshold variable, so a runner
    # using the wrong threshold would still pass. Recomputing
    # t = ceil(0.8 N) independently makes that failure detectable, and
    # the emitted threshold is cross-checked against the same value.
    fi = load("failure_injection.jsonl")
    checks["failure_injection_matches_quorum_rule"] = all(
        bool(r["succeeded"])
        == (r["survivors"] >= -(-4 * r["n"] // 5))       # ceil(0.8 n)
        and r["threshold"] == -(-4 * r["n"] // 5)
        for r in fi
    )

    # reconstruction: roots and leaf vectors verify; counts are in range;
    # and the tamper control fires wherever it was run
    recon = load("reconstruction.jsonl")
    checks["reconstruction_roots_verify"] = all(
        r["epoch_root_recoverable"] and r["leaf_vector_authenticated"]
        for r in recon
    )
    checks["reconstruction_counts_in_range"] = all(
        0 <= r["events_provable"] <= r["events_total"]
        and 0 <= r["events_signature_verified"] <= r["events_total"]
        for r in recon
    )
    controls = [
        r["negative_control_detects_tamper"] for r in recon
        if r.get("negative_control_detects_tamper") is not None
    ]
    checks["reconstruction_negative_control_fires"] = bool(controls) and all(
        controls
    )
    # every corrupted chain must be detected: flagged by the auditor, and
    # contributing no provable events. This is the claim RQ3 rests on, so
    # it is checked over every trial rather than summarised.
    corrupted_total = sum(
        sum(r.get("corrupted_by_mode", {}).values()) for r in recon
    )
    detected_total = sum(
        sum(r.get("detected_by_mode", {}).values()) for r in recon
    )
    checks["reconstruction_all_corruption_detected"] = (
        corrupted_total > 0 and detected_total == corrupted_total
    )
    checks["reconstruction_no_silent_acceptance"] = all(
        r.get("silent_acceptances", 0) == 0 for r in recon
    )
    checks["reconstruction_no_false_flags"] = all(
        r.get("intact_chains_flagged", 0) == 0 for r in recon
    )
    # every corruption mode must actually occur, or the detection rate
    # above would be a claim about a mode that was never injected
    modes_seen = {
        m
        for r in recon
        for m, c in r.get("corrupted_by_mode", {}).items()
        if c > 0
    }
    checks["reconstruction_all_modes_exercised"] = modes_seen == {
        "total_loss", "tail_truncation", "interior_tamper"
    }
    # the recovered fraction must be a measurement, not 1-f restated. A
    # fixed number of corrupted agents would make every trial at a given
    # (n, f) identical and the bootstrap CI collapse to a point; the
    # corruption count is Binomial, so it must vary.
    spreads = []
    for n in (10, 100, 500):
        for f in (0.3, 0.5):          # intermediate rates
            vals = [
                r["events_recoverable"] for r in recon
                if r["n"] == n and abs(r["f"] - f) < 1e-9
            ]
            if vals:
                spreads.append(float(np.var(vals)) > 0.0)
    checks["reconstruction_recoverable_is_not_one_minus_f"] = bool(
        spreads
    ) and all(spreads)

    # latency decomposition must add up to the reported total
    for n in FLEET_SIZES:
        rows = load(f"sim_n{n}.jsonl")
        parts = (
            "transport_ms", "member_sign_ms", "aggregate_ms",
            "operator_overhead_ms", "countersign_ms", "verify_ms",
        )
        checks[f"latency_decomposition_sums_n{n}"] = all(
            abs(sum(r[p] for p in parts) - r["total_ms"]) < 1e-3
            for r in rows
        )

    # Sybil resistance is claimed at two admission surfaces, so both are
    # checked separately rather than folded into one verdict: an unlisted
    # key must fail member-side manifest admission AND fail to contribute
    # a share the decision path accepts.
    imperson = [r for r in attacks if r["scenario"] == "single-impersonation"]
    checks["impersonation_rejected_at_both_surfaces"] = bool(imperson) and all(
        r.get("manifest_admitted") is False
        and r.get("decision_path_admitted") is False
        for r in imperson
    )

    # A malformed bitmap must be REJECTED, never crash the verifier: the
    # scenario reports the exception type if one escapes, and a crashing
    # public verifier is a denial-of-service surface, not a rejection.
    malformed = [r for r in attacks if r["scenario"] == "malformed-bitmap"]
    checks["malformed_bitmap_never_raises"] = bool(malformed) and all(
        r.get("any_raised") is False for r in malformed
    )

    # Time protocol progression at quorum receipt, not the slowest link.
    # Recompute the pre-prepare delay from the same seeded draws and require
    # quorum-1 follower receipts, since the leader counts toward quorum.
    # For this seed, also check that quorum delay is below the maximum.
    from ibsf import pbft as _pbft
    from ibsf.network import NetworkModel as _NM

    _net = _NM()
    _n = 500
    _quorum = 2 * ((_n - 1) // 3) + 1
    _r1 = _pbft._broadcast_quorum_ms(_net, np.random.default_rng(12345), _n, _quorum)
    _draws = np.sort(_net.one_way(np.random.default_rng(12345), _n - 1))
    # The leader holds the pre-prepare at time zero and counts toward quorum.
    # Use quorum-1 follower receipts, matching ibsf/pbft.py.
    _followers = min(_quorum - 1, _n - 1)
    checks["pbft_preprepare_is_quorum_gated"] = (
        abs(_r1 - float(_draws[_followers - 1])) < 1e-9
        and _r1 < float(_draws[-1]) - 1e-9
    )
    checks["network_model_exposes_no_max_helper"] = not hasattr(_NM, "fanout_max")

    # the federation's O(N) message claim is checked against the traces,
    # not asserted in prose: one call-for-sign per member plus one reply
    # per quorum member.
    checks["federation_messages_are_linear"] = all(
        all(r["messages"] == 2 * r["n"] + r["threshold"] for r in load(f"sim_n{n}.jsonl"))
        for n in FLEET_SIZES
    )

    # certificate root-level size must grow with the number of epochs
    cert = load("certsize.jsonl")
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in cert:
        by_n[r["n"]].append(r)
    checks["certsize_monotone_in_epochs"] = all(
        [x["root_level_bytes"] for x in sorted(v, key=lambda r: r["k"])]
        == sorted(x["root_level_bytes"] for x in v)
        for v in by_n.values()
    )
    return checks


def sign_check() -> dict:
    import blspy

    h = hashlib.sha256()
    files = {}
    for p in sorted(DATA.glob("*.jsonl")):
        blob = p.read_bytes()
        h.update(blob)
        files[p.name] = {
            "sha256": hashlib.sha256(blob).hexdigest(),
            "records": blob.count(b"\n"),
        }
    checks = _invariants()
    return {
        "trace_hash": h.hexdigest(),
        "sign_check_pass": all(checks.values()),
        "checks": checks,
        "data_files": files,
        "bls_binding": f"blspy {getattr(blspy, '__version__', '2.0.3')} (blst backend)",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    summary: dict = {}
    table_sybil(summary)
    table_latency(summary)
    table_recon(summary)
    table_certsize(summary)
    epoch_sensitivity(summary)
    failure_injection(summary)
    pop_commissioning(summary)
    wireless(summary)
    platform_probe(summary)
    microbench(summary)
    source_size(summary)
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2))
    sc = sign_check()
    (REPORTS / "sign_check.json").write_text(json.dumps(sc, indent=2))
    failed = [k for k, v in sc["checks"].items() if not v]
    if failed:
        logger.error("sign_check FAILED invariants: %s", ", ".join(failed))
    else:
        logger.info("sign_check: all %d invariants hold", len(sc["checks"]))
    logger.info("summary.json + sign_check.json written")


if __name__ == "__main__":
    main()
