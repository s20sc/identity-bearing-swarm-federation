# Operator-Governed Fleet Authorisation and Offline Audit for Heterogeneous Multi-Robot Systems

This repository contains the reference implementation and evaluation tools for authorising joint fleet decisions and verifying retained evidence offline.

The paper uses the fixed software-and-data release [software and data archive](https://doi.org/10.5281/zenodo.23551524). Download `Supplementary_code_and_data.zip` from that record to reproduce the paper. When using the development repository, record the Git commit alongside any regenerated outputs.

## Set up

From the extracted archive or a checkout, create a Python 3.10 or newer environment and install the dependencies declared in `pyproject.toml`.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## Recompute summaries from the recorded runs

```sh
python scripts/make_tables.py
```

This command aggregates the supplied trial records and platform measurements under `data/` and writes summaries, table fragments and consistency checks under `reports/`. The regenerated `reports/sign_check.json` records trial data hashes, software versions and the check results. The source-size utility describes the files in the copy where it runs. Use the fixed archive's files for source-size figures in the paper.

The `scenario` field in `data/attacks.jsonl` identifies six scripted checks and a separate operator-key-compromise control, with 120 trials per scenario. The roster-substitution control tests the countersignature's joint binding to the roster and participant bitmap. The seed-swept experiments use 30 seeds per configuration.

The reference implementation measures cryptographic operations on one host. Transport, scheduling and faults are modelled. The PBFT timing model charges HMAC-SHA256 tag generation at one endpoint and omits tag comparison.

### Reconstruction counts

Figure 5 and Table 4 use `data/reconstruction.jsonl`. The original sample size in each trial is 20 records per member.

| Field | Meaning |
|---|---|
| `events_total` | Original number of records in the trial |
| `events_present` | Records in the supplied logs |
| `events_signature_verified` | Records in supplied logs that pass the signature-only screening |
| `events_provable` | Records on chains admitted against their committed heads |
| `events_recoverable` | `events_provable / events_total`, rounded to six decimal places |

## Check the offline example

```sh
python scripts/verify_cert.py
```

The script uses the bundled certificate, event proof and operator public key under `data/verify_cert/`. It checks the signed epoch history, the roster and the event's signature, log suffix and fleet Merkle path. It exercises ten rejection controls. For deployment, obtain the operator key through a trusted channel and use an external freshness anchor to establish the currency of an exported history. Checking the original approval of a decision also requires the original decision and approval tuple.

The verification command reads the committed example. Supplying `--export` creates new example files before verification.

## Run new measurements

In a separate working copy, run `python scripts/run_all.py` to generate trial data, aggregate reports and check the verifier. This overwrites recorded outputs in that copy. Wall-clock times depend on the machine and software environment. Record the configuration and retain `reports/run_provenance.json`. The optional `--manuscript` check is tied to the TeX patterns declared in the numerical checker.

To run the regression suite, install `pytest` in the environment and run `python -m pytest` from the repository root.

## Cite and license

Cite the [fixed software-and-data release](https://doi.org/10.5281/zenodo.23551524) and record the Git commit for any development snapshot you use. The implementation is distributed under the [MIT license](https://github.com/s20sc/identity-bearing-swarm-federation/blob/main/LICENSE).
