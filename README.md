# EvoCompass

**Homolog-conditioned probabilistic protein editing**

EvoCompass is an event-driven protein sequence model conditioned on homolog-specific
evolutionary correspondence. At each position it combines the current residue, a
20-way MSA profile, and normalized time to predict:

1. whether an edit occurs;
2. whether it is a substitution, deletion, or insertion; and
3. the destination residue for substitutions and insertions.

Calibrated event probabilities are converted to continuous-time hazards and used to
sample variable-length editing trajectories. The compact heads are trained with
supervised event and residue losses, not a flow-matching objective.

## Why evolutionary correspondence?

A source sequence alone generally does not identify a particular evolutionary
direction. EvoCompass makes the homolog-specific amino-acid distribution an explicit
conditioning variable instead of asking an intermediate sequence state to implicitly
encode an unseen endpoint.

Controlled input perturbations show dependence on aligned homolog information.
The original prefix score evaluates requested edits from a fixed source state;
it is not a sequential trajectory likelihood. Family-wide profile shuffling also
changes branch provenance and can import leaves excluded for the recipient, so
it does not isolate position alone. The dose experiment interpolates each profile
vector between shuffled and true profiles rather than restoring a fraction of
correctly paired columns.

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

The compact aggregate reports in `artifacts/reports` preserve the original frozen
evaluation, not the complete current manuscript analysis:

- 15 held-out protein families;
- 1,557 reconstructed evolutionary branches;
- positive correspondence effects in all held-out families;
- a monotonic 0%, 25%, 50%, 75%, 100% correspondence dose response;
- increasing cumulative fixed-source prefix-score differences across edit budgets;
- improved homolog-supported editing and forward-direction likelihood.

The checkpoint historically named `evoflows_reproduction` is a supervised
baseline, not an EvoFlows reproduction: it uses source identity, position,
branch length, time, and zero padding with the same supervised head recipe.
Archived filenames and report labels are retained for provenance, but their
reproduction label is superseded by this correction. These artifacts do not
provide an empirical comparison with EvoFlows' flow-matching method.

## Limitations

- Reconstructed endpoints do not reveal the true historical edit order.
- Exact inserted-residue recovery is weaker than substitution identity prediction.
- The small reference heads do not model long-range sequence context or epistasis.
- Evolutionary plausibility is not equivalent to an engineering objective such as
  thermostability.
- Edit likelihood does not establish historical trajectories or improved function.

## Code and data availability

This repository provides the compact reference implementation, compact checkpoints,
and original aggregate reports. The complete manuscript reproduction package,
including benchmark inputs, alignment/tree caches, contextual feature caches, and
revision-specific analysis scripts, can be requested from the corresponding author
(mbuehler@mit.edu). The released artifacts alone are therefore insufficient to reproduce every manuscript result;
the original reports should not be interpreted as the corrected manuscript analyses.

## Relationship to DiscoverydLLM

EvoCompass was developed using the Edit Flow infrastructure in DiscoverydLLM. This
repository contains the protein-specific evolutionary conditioning, calibrated hazard
model, evaluation controls, and compact release artifacts.

