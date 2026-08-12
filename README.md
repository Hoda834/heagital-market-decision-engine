# Heagital Market Decision Engine

A parameterised decision system that translates fragmented NHS healthcare data into
auditable, ranked market entry decisions for a UK MedTech startup.

This is not a reporting or BI project. It encodes strategic trade-offs explicitly,
records exactly which parameters produced each ranking, and shows which decisions
survive a change of assumptions.

## What the engine produces

- A ranked rollout list of NHS Integrated Care Boards (ICBs)
- A recommended market entry cut-off (Top N)
- Scenario-adjusted rankings and a sensitivity table separating **robust** decisions
  from **fragile** ones
- A machine-readable audit record for every run

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"

# Rank the bundled ICB dataset
heagital-mde --scenarios

# Or run the interactive app
streamlit run app.py
```

Without an editable install, invoke the CLI with the source directory on the path:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m heagital_mde.cli.run --scenarios
```

Run the tests with `pytest`.

## Inputs

ICB-level features, one row per ICB. Download `data/template/icb_input_template.csv`
for a filled example.

| Column | Required | Meaning |
| --- | --- | --- |
| `ICB ODS code` | yes | Unique ICB identifier |
| `ICB name` | yes | Display name |
| `Register` | yes | AF register size |
| `Prevalence (%)` | yes | AF prevalence, 0–100 |
| `Treatment Gap (%)` | yes | Share of the register not anticoagulated |
| `Warfarin Item icb` | yes | Warfarin prescribing proxy |
| `Region` | no | NHS England region, used for regional rollups and the map |
| `Digital Maturity (0-1)` | no | Adoption readiness signal |
| `Procurement Friction (0-1)` | no | Friction penalty signal |

Headers are matched case-insensitively, and thousands separators and `%` signs are
parsed. Extra columns in the file are ignored.

**Treatment gap units.** The column is read as a percentage by default. Files that
already hold fractions are detected automatically; use `--gap-units fraction` to be
explicit. This is a real source of silent 100× errors, so the engine tells you which
interpretation it applied in the audit record.

## Decision logic

```
final = alpha × market + (1 − alpha) × readiness − friction_weight × friction
```

| Pillar | Built from | Notes |
| --- | --- | --- |
| `market` | register, prevalence, treatment gap, warfarin proxy | Addressable opportunity |
| `readiness` | treatment gap, warfarin proxy, digital maturity | How easily an ICB can adopt |
| `friction` | procurement friction | A penalty, applied outside the blend |

Every signal is normalised to 0–1 before weighting. Weights within each pillar are
renormalised to sum to 1, so they express relative emphasis rather than absolute
scale. `alpha` and `friction_weight` are the two strategic levers: `alpha = 1.0`
ranks purely on market size, `alpha = 0.0` purely on readiness.

### Known modelling limitation

`readiness` and `market` share two of their three signals. On the bundled dataset
their scores correlate at roughly **0.88**, which means `alpha` has limited power to
reorder the list. The engine measures this correlation on every run and warns when
it exceeds 0.85.

The fix is data, not code: supply a `Digital Maturity (0-1)` column so readiness has
an independent basis. Until then, treat `alpha` as a sensitivity knob rather than a
genuine two-dimensional trade-off.

`register` and `prevalence` also overlap by construction (register ≈ prevalence ×
population), so weighting both emphasises size twice.

## Configuration

`src/heagital_mde/config/scoring_config.yml` holds the base parameters. Every
recognised key is applied and **every unrecognised key raises an error** — a typo can
never silently fall back to built-in defaults.

`normalisation.method` accepts:

- `minmax` — preserves magnitude, sensitive to outliers
- `rank` — percentile position, far more stable on a 42-row panel

Set `normalisation.winsorize` (e.g. `0.05`) to clamp tails before min-max scaling.
A signal with no spread scores `constant_fill` (0.5 by default) rather than 0, so it
stays neutral instead of silently forfeiting its weight.

## Scenarios and sensitivity

`src/heagital_mde/config/scenarios.yml` defines strategic scenarios as *deltas* on the
base configuration — `nice_tightening`, `warfarin_decline`,
`community_testing_expansion`, `procurement_squeeze`, plus `base_case`.

`heagital-mde --scenarios` scores the data under each and writes
`data/outputs/rankings/icb_sensitivity.csv`, classifying every ICB as:

- **robust** — inside the cut-off under every scenario
- **fragile** — inclusion depends on which scenario holds
- **excluded** — outside the cut-off under every scenario

The fragile group is the only one where the strategic assumption actually changes the
decision, and is where review effort belongs.

## Auditability

Every CLI run writes `run_audit.json` alongside the ranking, recording the engine
version, a UTC timestamp, the input file's SHA-256, the fully resolved weights
actually applied, which optional signals were present or absent, how many rows were
dropped, the market/readiness correlation, and every warning raised. The same record
is shown in the app's **Audit** tab.

## Validation

Input problems are collected and reported together, not one at a time. The engine
rejects duplicate ICB codes, blank identifiers, negative counts, prevalence above
100%, and treatment gaps outside 0–1; it warns on blank regions and on panels too
small for normalisation to spread scores meaningfully. Use `--strict` to treat
warnings as errors.

## Outputs

Written to `data/outputs/rankings/`:

| File | Contents |
| --- | --- |
| `icb_opportunity_ranking_basecase.csv` | Ranked ICBs with pillar scores and cut-off flags |
| `run_audit.json` | Full reproduction record |
| `icb_sensitivity.csv` | Rank stability across scenarios (`--scenarios`) |
| `scenarios/icb_ranking_<name>.csv` | Per-scenario ranking (`--scenarios`) |

## Project layout

```
app.py                              Streamlit interface
src/heagital_mde/
  cli/run.py                        Command-line entry point
  config/scoring_config.yml         Base parameters
  config/scenarios.yml              Strategic scenarios
  data/schema.py                    Column contract and header resolution
  io/load_icb.py                    Parsing and canonicalisation
  io/validate.py                    Input validation
  model/normalise.py                Normalisation strategies
  model/scenarios.py                Scenario application and sensitivity
  model/scoring/                    Market, readiness, friction, blend, rank
tests/                              Test suite
```

## Map data

The regional map uses NHS England region boundaries from
`data/geo/nhs_england_regions.geojson`. That file currently ships empty, so the app
falls back to region centroids. Drop in a populated GeoJSON to switch to boundary
shading automatically — no code change needed.
