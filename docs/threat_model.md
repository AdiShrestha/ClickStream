# Clickstream Threat Model and System Design (v1, Week 1)

## System under study

A continuous behavioral-authentication system for banking sessions,
using keystroke dynamics (primary modality, Weeks 1-7) and mouse
dynamics (secondary modality, introduced from Week 2 onward as time
permits), scoring live session behavior against a per-user enrolled
baseline via classical (Isolation Forest / One-Class SVM) and
deep-sequence (Siamese network) models.

## Research questions

- RQ1: How much does a Frog-Boiling-style gradual poisoning attack
  degrade a continuously-adapting authentication baseline, and does a
  residue-feature/RONI-style validation gate measurably mitigate it
  without blocking legitimate behavioral drift?
- RQ2: How effective is a video-based (screen-recording) injection
  attack against this system's keystroke-timing verification, and does
  a secondary liveness signal or synthetic-timing detector measurably
  reduce its evasion rate?
- RQ3 (stretch): How does the poisoning threat model change under
  federated learning, and does Byzantine-robust aggregation (Krum /
  trimmed mean) mitigate a client-level version of the same attack
  where naive FedAvg does not?

## Threat model

- Poisoning attacker (RQ1): controls sessions that get absorbed into
  the baseline-adaptation loop (e.g., via a low-risk-scored stolen
  session); cannot directly modify model code or infrastructure.
- Injection attacker (RQ2): possesses a screen recording of the victim
  typing and can inject synthetic timing data into the pipeline; does
  not need a keylogger or physical device access.
- Federated attacker (RQ3, stretch): controls one or more federated
  client devices and can send arbitrary but plausible-looking updates.
- Out of scope: network/TLS-layer attacks, sensor-level presentation
  attacks (ISO 30107 territory), physical device compromise beyond
  what is stated above.

## Explicit scope boundaries

- All quantitative results come from public academic benchmark
  datasets, not real banking telemetry.
- Proposed defenses are evaluated empirically, not proven unbreakable.
- Both attacks are extensions of published work, not novel attack
  classes invented from scratch.
- RQ3 is an explicitly secondary/stretch contribution.

## Datasets

- CMU Keystroke Dynamics Benchmark (Killourhy and Maxion, 2009) --
  primary, acquired Week 1.
- Balabit Mouse Dynamics Challenge -- secondary modality, from Week 2.
- Self-recorded video (author only, single-subject proof-of-concept,
  informed self-consent) -- Weeks 6-7 injection-attack demo only.

## Changelog

- Week 1: initial draft.
