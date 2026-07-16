# ADR 0001: operator and candidate key result directories

- Status: accepted
- Date: 2026-07-16

## Decision

Use `results/<operator_id>/<candidate_id>/<evaluation_id>/` as the physical
result location. Keep `run_id` as correlation metadata only.

## Rationale

The first two result components now mirror the candidate source tree exactly.
This makes ownership, source identity, repeated evaluations, retention, and
lookup unambiguous. Source changes require a new hash-backed candidate ID;
environment or invocation changes create a new evaluation below that candidate.

## Consequences

All path construction goes through validated mirror helpers. No component may
create a `results/<run_id>` result root. A future run index can point to several
operator/candidate/evaluation directories without changing their ownership.
