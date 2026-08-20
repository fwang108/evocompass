# EvoCompass

**Correspondence-Guided Edit Flows for Directional Protein Evolution**

EvoCompass is an event-driven protein sequence model conditioned on homolog-specific
evolutionary correspondence. At each position it combines the current residue, a
20-way MSA profile, and normalized time to predict:

1. whether an edit occurs;
2. whether it is a substitution, deletion, or insertion; and
3. the destination residue for substitutions and insertions.

Calibrated event probabilities are converted to continuous-time hazards and used to
sample variable-length editing trajectories.

## Why evolutionary correspondence?

A source sequence alone generally does not identify a particular evolutionary
direction. EvoCompass makes the homolog-specific amino-acid distribution an explicit
conditioning variable instead of asking an intermediate sequence state to implicitly
encode an unseen endpoint.

Controlled evaluations show that the benefit depends on correspondence, rather than
merely receiving more profile-like features. Destroying positional correspondence,
using an unrelated family, or replacing the profile with matched random information
degrades trajectory likelihood. Progressively restoring correspondence produces a
monotonic improvement.

## Model

For each operation, EvoCompass uses separate calibrated event and identity heads:

```text
source residue (20) + MSA profile (20) + normalized time (1)
                              |
                   +----------+----------+
                   |                     |
              event head            identity head
                   |                     |
             calibrated p(edit)      p(residue | edit)
                   |
              CTMC hazard
```

The released reference has 30,303 neural parameters across the three operations,
plus six fitted calibration scalars. It is deliberately small so that evolutionary
conditioning can be isolated from backbone capacity.

## Installation

```bash
cd evocompass
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,analysis]"
pytest
```

## Data contract

Input data are JSON Lines records with this schema:

```json
{
  "family": "PF00000",
  "group": "branch-id",
  "operation": "substitution",
  "source": "A",
  "msa_profile": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
  "time": 0.5,
  "event": 1,
  "target": "V"
}
```

The full benchmark is not committed because it is reconstructed from external
protein-family alignments and phylogenies. Descendant clades must be excluded from
the conditioning MSA before profile construction. Family-disjoint splits are
required for evaluation.

Convert rows into operation-specific arrays:

```bash
evocompass-prepare transitions.jsonl data/processed
```

The split file is JSON with `train`, `validation`, and `test` family-ID arrays.

## Training

```bash
evocompass-train \
  data/processed \
  split.json \
  outputs/evocompass.pt \
  --device cuda
```

## Evaluation

```bash
evocompass-evaluate \
  outputs/evocompass.pt \
  data/processed \
  split.json \
  --output outputs/evaluation.json
```

Statistical comparisons should use protein families, not positions or branches, as
the inferential unit. The reference protocol uses 10,000 family-bootstrap draws and
20,000 sign-flip permutations.

## Sampling

`evocompass-sample` accepts a JSON request containing a sequence, normalized time,
and candidate positions with MSA profiles. It converts calibrated event probabilities
to hazards, samples a waiting time and edit, and emits the edited sequence.

```bash
evocompass-sample artifacts/checkpoints/evocompass.pt request.json --seed 7
```

## Main results

The compact aggregate reports in `artifacts/reports` record the frozen evaluation:

- 15 held-out protein families;
- 1,557 reconstructed evolutionary branches;
- positive correspondence effects in all held-out families;
- a monotonic 0%, 25%, 50%, 75%, 100% correspondence dose response;
- increasing advantage over multi-step trajectories;
- improved homolog-supported editing and forward-direction likelihood.

The included EvoFlows comparator is an independent paper-contract reproduction. No
official implementation or checkpoint was publicly available, so it must not be
interpreted as evaluation of the authors' released model.

## Limitations

- Reconstructed endpoints do not reveal the true historical edit order.
- Exact inserted-residue recovery is weaker than substitution identity prediction.
- The small reference heads do not model long-range sequence context or epistasis.
- Evolutionary plausibility is not equivalent to an engineering objective such as
  thermostability.
- The benchmark supports directional trajectory modeling, not reconstruction of the
  unique historical trajectory.

## Relationship to DiscoverydLLM

EvoCompass was developed using the Edit Flow infrastructure in DiscoverydLLM. This
repository contains the protein-specific evolutionary conditioning, calibrated hazard
model, evaluation controls, and compact release artifacts.

