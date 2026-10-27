"""Check declared manuscript quantities against generated reports.

Each quantity pairs a manuscript pattern with its expected value from
summary.json. The checks cover selected table entries and running prose,
including their rounding rules. Quantities not declared here are not
checked. Compilation and experiment tests alone do not validate prose.

    python3 scripts/check_manuscript_numbers.py /path/to/manuscript.tex

Exit status is 1 on any mismatch, allowing use as a build gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports" / "summary.json"


def rounded(value: float, places: int) -> str:
    """Format the way the manuscript does, so comparison is like-for-like."""
    return f"{value:.{places}f}"


def checks(s: dict) -> list[tuple[str, str, str]]:
    """(label, expected string, regex that must find it in the tex).

    The regex anchors on surrounding words rather than on the bare
    number, so a match confirms the right quantity was found and not a
    coincidentally equal figure elsewhere in the paper.
    """
    out: list[tuple[str, str, str]] = []

    fed = [s["latency"][f"n{n}"] for n in (10, 50, 100, 500)]
    pbft = [s["pbft"][f"n{n}"] for n in (10, 50, 100, 500)]

    # Table IV row: medians and p99 for both protocols, one decimal
    fed_cells = " & ".join(
        f"{rounded(x['median_ms'],1)} [{rounded(x['p99_ms'],1)}]" for x in fed
    )
    pbft_cells = " & ".join(
        f"{rounded(x['median_ms'],1)} [{rounded(x['p99_ms'],1)}]" for x in pbft
    )
    out.append(("Table IV federation row", fed_cells,
                r"Federation \(this work\)\s*&\s*(.+?)\s*\\\\"))
    out.append(("Table IV PBFT row", pbft_cells,
                r"PBFT baseline\s*&\s*(.+?)\s*\\\\"))

    # abstract and body medians at the extremes
    out.append(("abstract N=10 median", rounded(fed[0]["median_ms"], 1),
                r"decision latency is ([\d.]+)\\,ms at \$N\{=\}10\$"))
    out.append(("abstract N=500 median", rounded(fed[3]["median_ms"], 1),
                r"\n([\d.]+)\\,ms at \$N\{=\}500\$ under"))
    out.append(("VIII-B range low", rounded(fed[0]["median_ms"], 1),
                r"Median latency runs from ([\d.]+)\\,ms"))
    out.append(("VIII-B range high", rounded(fed[3]["median_ms"], 1),
                r"to ([\d.]+)\\,ms at \$N\{=\}500\$ \(CI"))
    out.append(("VIII-B p99 low", rounded(fed[0]["p99_ms"], 1),
                r"\$p99\$ tail from ([\d.]+) to"))
    out.append(("VIII-B p99 high", rounded(fed[3]["p99_ms"], 1),
                r"\$p99\$ tail from [\d.]+ to ([\d.]+)\\,ms"))

    # wireless
    w = s["wireless"]
    out.append(("wireless N=100 median", rounded(w["n100"]["median_ms"], 1),
                r"becomes ([\d.]+)\\,ms at \$N\{=\}100\$"))
    out.append(("wireless N=500 median", rounded(w["n500"]["median_ms"], 1),
                r"and ([\d.]+)\\,ms at \$N\{=\}500\$ \(\$p99\$"))
    out.append(("wireless p99 pair",
                f"{rounded(w['n100']['p99_ms'],1)} and "
                f"{rounded(w['n500']['p99_ms'],1)}",
                r"\(\$p99\$ ([\d.]+ and [\d.]+)\\,ms\)"))

    # regulator verification time for the largest certificate, in seconds
    verify_s = max(
        s["certsize"][f"n{n}/k1000"]["regulator_verify_ms"] for n in (10, 100, 500)
    ) / 1000.0
    out.append(("regulator verify (1000 epochs)", rounded(verify_s, 1),
                r"offline in about\s*\n?([\d.]+)\\,s"))

    # certificate root size per epoch and at k=1000
    root_1000 = s["certsize"]["n10/k1000"]["root_level_bytes"]
    out.append(("cert root k=1000 (kB)", str(round(root_1000 / 1000)),
                r"1000-epoch root is ([\d]+)\\,kB"))

    # Check the certificate-size table byte-exactly. The three fleet-size
    # rows share the same root size because roots have no per-member field.
    def _grp(n):  # LaTeX-grouped thousands, e.g. 50{,}660
        return f"{n:,}".replace(",", "{,}")
    cz = s["certsize"]
    cert_cols = " & ".join(_grp(cz[f"n10/k{k}"]["root_level_bytes"])
                           for k in (1, 10, 100, 1000))
    # Anchor on the first column value (the per-epoch root, grouped), which
    # only the cert-size table carries, so this does not match a latency or
    # Sybil row that also begins "$N{=}10$ &".
    first_col = _grp(cz["n10/k1"]["root_level_bytes"])
    out.append(("Table VIII cert-size row", cert_cols,
                r"\$N\{=\}10\$\s*&\s*(" + re.escape(first_col)
                + r"\s*&[^\\]+?)\\\\"))

    # Full-document sizes, including root chain and deduplicated roster,
    # at the two fleet-size extremes in kB.
    fd10 = round(cz["n10/k1000"]["full_doc_bytes"] / 1000)
    fd500 = round(cz["n500/k1000"]["full_doc_bytes"] / 1000)
    out.append(("full-doc N=10 (kB)", str(fd10),
                r"full 1000-epoch document is ([\d]+)\\,kB at \$N\{=\}10\$"))
    out.append(("full-doc N=500 (kB)", str(fd500),
                r"at \$N\{=\}10\$ and\s*\n?([\d]+)\\,kB at \$N\{=\}500\$"))

    # The charge-both accounting prints the federation medians again; they
    # must equal the as-reported medians (charging both leaves the
    # federation unchanged). Check both occurrences against the same data.
    out.append(("charge-both federation row",
                ", ".join(rounded(x["median_ms"], 1) for x in fed[:3])
                + " and " + rounded(fed[3]["median_ms"], 1),
                r"against the federation's\s*\n?"
                r"([\d.]+, [\d.]+, [\d.]+ and [\d.]+)\\,ms"))

    # Check all three scheduling accountings against make_tables output.
    acc = s.get("scheduling_accountings", {}).get("per_fleet_size", {})
    if acc:
        bare = [acc[f"n{n}"]["federation_without_scheduling_ms"]
                for n in (10, 50, 100, 500)]
        out.append(("no-scheduling federation range",
                    f"{rounded(min(bare),1)}--{rounded(max(bare),1)}",
                    r"leaves the federation at ([\d.]+--[\d.]+)\\,ms"))
        withl = [acc[f"n{n}"]["pbft_with_leader_scheduling_ms"]
                 for n in (10, 50, 100, 500)]
        out.append(("PBFT with leader scheduling",
                    ", ".join(rounded(x, 1) for x in withl[:3])
                    + " and " + rounded(withl[3], 1),
                    r"puts PBFT at ([\d.]+, [\d.]+, [\d.]+\s*\n?and [\d.]+)\\,ms"))

    # Check the VIII-F extrapolation from the measured cost decomposition.
    c = s["latency"]["n500"]["components_median_ms"]
    crypto = (c["member_sign_ms"] + c["aggregate_ms"]
              + c["countersign_ms"] + c["verify_ms"])
    other = c["transport_ms"] + c["operator_overhead_ms"]
    out.append(("VIII-F crypto subtotal", rounded(crypto, 2),
                r"decomposes into ([\d.]+)\\,ms of cryptography"))
    out.append(("VIII-F non-crypto subtotal", rounded(other, 2),
                r"and ([\d.]+)\\,ms that is not"))
    out.append(("VIII-F 10x extrapolation", rounded(crypto * 10 + other, 1),
                r"gives \$[\d.]+ \+ [\d.]+ =\s*\n?([\d.]+)\$\\,ms"))

    # The RQ2 component decomposition (§8.3) quotes verification separately
    # from VIII-F; both must equal verify_ms. One read 2.5, VIII-F read 2.7.
    out.append(("RQ2 verification component", rounded(c["verify_ms"], 1),
                r"scheduling\s*\n?[\d.]+\\,ms, verification ([\d.]+)\\,ms"))

    # message counts
    out.append(("federation messages N=500",
                f"{s['latency']['n500']['messages']:,}".replace(",", "{,}"),
                r"\$2N\{\+\}t \\leq 3N\$ \(([\d{},]+) at \$N\{=\}500\$\)"))

    # Check the platform table and its slowdown band using unrounded
    # measurements from the generated summary, then apply display rounding.
    pp = s.get("platform_probe", {})
    plats = pp.get("platforms", {})
    ref_label = pp.get("reference")
    others = [k for k in plats if k != ref_label]
    if ref_label and len(others) == 1:
        ref_m = plats[ref_label]["median_ms"]
        oth_m = plats[others[0]]["median_ms"]
        band = plats[others[0]]["slowdown_vs_reference"]
        out.append(("VIII-F slowdown band min", rounded(band["min"], 2),
                    r"slows by nearly the same factor, between \$([\d.]+)\$"))
        out.append(("VIII-F slowdown band max", rounded(band["max"], 2),
                    r"between \$[\d.]+\$ and\s*\n?\$([\d.]+)\$ with a median"))
        out.append(("VIII-F slowdown band median", rounded(band["median"], 2),
                    r"with a median of \$([\d.]+)\$"))
        # Eight table rows, ratio column, in the order the manuscript lists
        # them. Each ratio is recomputed from full-precision medians.
        table_keys = [
            ("Member signature", "member_sign/None"),
            ("Single verification", "single_verify/None"),
            ("PoP generation", "pop_prove/None"),
            ("PoP verification", "pop_verify/None"),
            (r"Aggregate verify, \$N\{=\}10\$", "fast_aggregate_verify/10"),
            (r"Aggregate verify, \$N\{=\}100\$", "fast_aggregate_verify/100"),
            (r"Aggregate verify, \$N\{=\}500\$", "fast_aggregate_verify/500"),
            (r"Aggregation, \$N\{=\}500\$", "aggregate/500"),
        ]
        for label, key in table_keys:
            ratio = oth_m[key] / ref_m[key]
            out.append((f"Table IX ratio: {key}", rounded(ratio, 2),
                        label + r"\s*&[^&]+&[^&]+&\s*([\d.]+)\s*\\\\"))

    # Prose quantities that live only in running text. These were the last
    # numbers outside the gate, and every one of them had been truncated
    # rather than rounded (0.1773 quoted as 0.17), which is precisely the
    # drift a gate exists to catch: the appendix claims every reported
    # number regenerates from the artifact, so the ones in prose have to
    # be declared here too.
    mb = s.get("microbench")
    if mb:
        out.append((
            "VIII-E key-list-digest micro-benchmark",
            rounded(mb["key_list_digest/n500"]["median_ms"], 2),
            r"serialises all \$N\$ roster keys \(([\d.]+)\\,ms at",
        ))
        out.append((
            "VIII-E distinct-key micro-benchmark",
            rounded(mb["distinct_key_count/n500"]["median_ms"], 2),
            r"participant keys \(([\d.]+)\\,ms\)",
        ))

    fi = s.get("failure_injection")
    if fi:
        for label, n, c, pat in [
            ("N=500 c=0.0", "n500", "c0.0",
             r"through \$c \\leq 0\.1\$ \(([\d.]+) to"),
            ("N=500 c=0.1", "n500", "c0.1",
             r"through \$c \\leq 0\.1\$ \([\d.]+ to ([\d.]+)\\,ms\)"),
            ("N=500 c=0.2", "n500", "c0.2",
             r"rises to ([\d.]+)\\,ms at \$c\{=\}0\.2\$"),
            ("N=100 c=0.0", "n100", "c0.0",
             r"same shape \(([\d.]+) to"),
            ("N=100 c=0.2", "n100", "c0.2",
             r"same shape \([\d.]+ to ([\d.]+)\\,ms\)"),
        ]:
            out.append((f"VIII-G liveness median {label}",
                        rounded(fi[f"{n}/{c}"]["median_ms"], 1), pat))

    # Check source-size counts reported in the manuscript prose.
    src = s.get("source_size")
    if src:
        def kloc(n):
            return f"{n / 1000:.1f}"
        out.append(("impl total kLOC", kloc(src["total"]),
                    r"implementation totals ([\d.]+)\\,kLOC"))
        out.append(("impl core kLOC", kloc(src["core"]),
                    r"([\d.]+)\\,kLOC\s*\n?in the core package"))
        out.append(("impl harness kLOC", kloc(src["harness"]),
                    r"([\d.]+)\\,kLOC in the evaluation harness"))
        out.append(("impl tests kLOC", kloc(src["tests"]),
                    r"([\d.]+)\\,kLOC of security-property"))
        out.append(("impl every-line kLOC", kloc(src["every_line"]),
                    r"or ([\d.]+)\\,kLOC\s*\n?counting every line"))

    pop = s.get("pop_commissioning")
    if pop:
        out.append((
            "VIII-G PoP generation",
            rounded(pop["prove_ms_median"], 2),
            r"costs ([\d.]+)\\,ms to generate on",
        ))
        out.append((
            "VIII-G PoP verification",
            rounded(pop["verify_ms_median"], 2),
            r"and ([\d.]+)\\,ms to verify at the operator",
        ))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manuscript", type=Path, nargs="?")
    ap.add_argument(
        "--glob",
        help="resolve the manuscript by pattern instead of by name, so a "
             "version bump cannot silently leave this check validating the "
             "previous document. Aborts unless the pattern matches exactly "
             "one file.",
    )
    args = ap.parse_args()

    if args.glob:
        matches = sorted((ROOT / args.glob).parent.glob(Path(args.glob).name))
        if len(matches) != 1:
            print(
                f"FAIL: --glob {args.glob!r} matched {len(matches)} files "
                f"{[m.name for m in matches]}; expected exactly one"
            )
            return 1
        args.manuscript = matches[0]
        print(f"checking {args.manuscript.name}")
    elif args.manuscript is None:
        print("FAIL: give a manuscript path or --glob")
        return 1

    if not SUMMARY.exists():
        print(f"FAIL: {SUMMARY} missing; run make_tables.py first")
        return 1
    s = json.loads(SUMMARY.read_text())
    tex = args.manuscript.read_text()

    bad = 0
    unmatched = 0
    for label, expected, pattern in checks(s):
        m = re.search(pattern, tex, re.S)
        if not m:
            print(f"UNMATCHED  {label:34s} pattern found nothing "
                  f"(prose reworded? update this check)")
            unmatched += 1
            continue
        found = " ".join(m.group(1).split())
        if found != expected:
            print(f"MISMATCH   {label:34s} manuscript={found!r} "
                  f"artifact={expected!r}")
            bad += 1

    total = len(checks(s))
    ok = total - bad - unmatched
    print(f"\n{ok}/{total} agree, {bad} mismatched, {unmatched} unmatched")
    if bad or unmatched:
        print("The appendix promises every number regenerates from the "
              "traces. Fix the manuscript, or the check, before compiling.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
