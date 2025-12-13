# heagital-market-decision-engine
It is a parameterised decision system designed to translate fragmented healthcare data into auditable, ranked market entry decisions.
# Heagital Market Decision Engine

This repository implements a decision-grade market entry engine for a UK MedTech startup operating in the NHS context.

This is not a reporting or BI project.
It is a parameterised decision system designed to translate fragmented healthcare data into auditable, ranked market entry decisions.

## What this system does

The engine evaluates NHS Integrated Care Boards (ICBs) using a weighted opportunity model that explicitly encodes strategic trade-offs.

It produces:
- A ranked rollout list of ICBs
- A recommended market entry cut-off
- Sensitivity signals showing which regions are robust or fragile to strategic change

## Inputs

ICB-level quantitative features, including:
- AF register size
- Treatment gap
- Warfarin prescribing proxy
- Population aged 65+
- Testing intensity indicators

Inputs are treated as state-of-world data, not assumptions.

## Decision parameters

The decision-maker defines three weights:

- Clinical Risk Weight (W1)
- Adoption Readiness Weight (W2)
- Procurement Friction Weight (W3)

Constraint:
W1 + W2 + W3 = 1

These parameters represent strategic intent, not statistical tuning.

## Core decision logic

OpportunityScore(ICB) =
W1 × Normalised(ClinicalRisk)
+ W2 × Normalised(AdoptionReadiness)
− W3 × Normalised(ProcurementFriction)

## Outputs

- Ranked ICB rollout list
- Top-N recommendation
- Scenario-adjusted rankings

## Quick start

```bash
pip install -r requirements.txt
python -m heagital_mde.cli.run
