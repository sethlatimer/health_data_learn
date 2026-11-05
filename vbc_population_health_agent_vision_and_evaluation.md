# VBC and Population Health Evidence Agent

## Purpose

Build a local evidence and analytics system that helps users answer value-based care (VBC) and population health questions by combining:

- Public healthcare datasets
- Academic literature
- Local documents
- Semantic search
- Deterministic Python/pandas analysis
- Local LLM interpretation and presentation

The system should help users move from a messy healthcare problem to an evidence-backed decision, intervention idea, or testable hypothesis. It should preserve provenance, distinguish facts from inference, and never rely on the LLM to manufacture data or calculate statistics.

## North-Star Workflow

```text
Question
  -> Public data and literature discovery
  -> Relevant dataset, measure, and population selection
  -> Bounded authoritative data retrieval
  -> Deterministic analysis
  -> Evidence synthesis
  -> VBC or community-health hypothesis
  -> Intervention options
  -> Internal analysis or evaluation design
  -> Decision and ongoing evaluation
```

## Core User Personas

### 1. Population Health Analyst: Where is the opportunity?

**Typical workflow**

- Starts with a broad problem, such as high avoidable admissions or poor primary-care access.
- Searches public datasets across geography, condition, utilization, demographics, provider supply, and community characteristics.
- Compares rates and trends.
- Searches literature for plausible drivers and interventions.
- Develops hypotheses to test with internal data.

**Desired outcomes**

- Identify a specific population, geography, or condition with a defensible opportunity.
- Separate utilization, price, care intensity, quality, access, and risk.
- Produce a prioritized list of questions worth investigating internally.

### 2. VBC Strategist or Program Designer: What should we do about it?

**Typical workflow**

- Begins with a defined problem, such as high COPD admissions or limited rural access.
- Searches academic evidence for interventions such as home-based care, care management, pharmacy programs, remote monitoring, community health workers, transportation, or food interventions.
- Compares alternative operating models and payment mechanisms.
- Identifies implementation requirements, risks, and evidence gaps.

**Desired outcomes**

- Generate several evidence-informed intervention options.
- Explain the expected clinical, utilization, and financial mechanism for each option.
- Clarify who pays, who benefits, what capabilities are required, and how success would be measured.
- Avoid presenting promising ideas as proven solutions.

### 3. Program Evaluator: Did the intervention actually work?

**Typical workflow**

- Starts with an existing program or intervention.
- Finds external benchmarks and published evidence.
- Defines expected outcomes, comparison groups, follow-up periods, and sensitivity analyses.
- Challenges alternative explanations such as selection bias, regression to the mean, secular trends, claims lag, coding differences, and price changes.
- Uses internal data for the evaluation and public evidence for context.

**Desired outcomes**

- Produce a defensible evaluation rather than an ROI victory lap.
- Clearly separate association from causation.
- Improve cohort design, measures, risk adjustment, and causal methods.
- Identify what can be concluded, what cannot, and what evidence is still needed.

### 4. VBC Leader: Where should we place our bets?

**Typical workflow**

- Asks decision-oriented questions rather than browsing datasets directly.
- Reviews concise opportunity summaries, intervention options, economics, risks, and evidence gaps.
- Drills into provenance or detailed analysis when an opportunity appears actionable.

**Desired outcomes**

- Receive a concise strategy brief with:
  - Opportunity and affected population
  - Scale and significance
  - Supporting and conflicting evidence
  - Intervention options
  - Financial and operational dependencies
  - Equity considerations
  - Unknowns and recommended next analysis

### 5. Curious Analyst or Evidence Explorer: What are we missing?

**Typical workflow**

- Explores multiple datasets and research sources without beginning with a fixed hypothesis.
- Looks for unusual patterns, gaps, inconsistencies, and underused interventions.
- Connects burden, access, utilization, outcomes, social factors, and adoption.
- Generates new questions for internal investigation.

**Desired outcomes**

- Discover a new question worth asking.
- Identify evidence-backed opportunities that were not already on the roadmap.
- Generate creative but testable VBC and community-health hypotheses.

## Evaluation Framework

### 1. Catalog and Source Coverage

Tests whether the system can find relevant public datasets and literature.

**Example handles**

- `CAT-001`: Find datasets about preventable inpatient utilization.
- `CAT-002`: Find datasets about rural primary-care access.
- `CAT-003`: Find literature about home-based care and readmissions.
- `CAT-004`: Find evidence about food insecurity interventions.

**Key metrics**

- Recall@10
- Precision@5
- Identifier accuracy
- Duplicate/version rate
- Metadata completeness

### 2. Semantic Retrieval

Tests whether ordinary VBC language maps to official dataset and research terminology.

**Example handles**

- `SEM-001`: "Avoidable hospitalizations"
- `SEM-002`: "People not getting primary care"
- `SEM-003`: "Doctors available in rural communities"
- `SEM-004`: "Diabetes complications that primary care could prevent"
- `SEM-005`: "Care shifting from the hospital to the home"

**Key metrics**

- Correct source in top 5
- Correct dataset in top 5
- Correct measure or concept in top 5
- Mean reciprocal rank
- False-positive rate

### 3. Data Query Correctness

Tests whether the system constructs a valid, bounded query using authoritative source metadata.

**Example handles**

- `DAT-001`: Asthma admissions in Australia, 2016-2023.
- `DAT-002`: Hospital beds per population in Japan.
- `DAT-003`: Physician supply in Canada and the United States.
- `DAT-004`: A CMS payment measure for a specified year and provider type.
- `DAT-005`: Request "last year" and resolve the latest period actually available.

**Required checks**

- Correct source and dataset
- Correct measure and unit
- Correct geography and subgroup filters
- Correct time window
- Correct dimension ordering
- Bounded row count
- No silent substitution of an unrelated measure or population

### 4. Numerical Correctness

Tests whether deterministic analysis reproduces validated answers.

**Current gold test**

```text
Australia asthma hospital admissions
Period: 2016-2023
First value: 70.4
Last value: 52.0
Absolute change: -18.4
Percent change: -26.14%
Mean: 55.5
```

**Additional edge cases**

- Zero baseline
- Missing or duplicate periods
- Suppressed values
- Revised observations
- Counts versus rates
- Crude versus age-standardized rates
- Nominal versus inflation-adjusted spending
- PMPM versus total spending

### 5. Literature and Evidence Quality

Tests whether the system understands research design and evidence strength.

**Example handles**

- `LIT-001`: Find systematic reviews on hospital-at-home outcomes.
- `LIT-002`: Find causal studies of community health workers and utilization.
- `LIT-003`: Find evidence on food-as-medicine programs.
- `LIT-004`: Find evidence supporting and contradicting remote monitoring.
- `LIT-005`: Distinguish trials, observational studies, reviews, and commentaries.

**Required metadata**

- PMID, PMCID, DOI, or another stable identifier
- Title, authors, journal, and publication year
- Study design and population
- Intervention, comparator, and outcomes
- Sample size when available
- Abstract or licensed full-text status
- Retraction or correction status
- Retrieval date and provenance

### 6. VBC Reasoning

Tests whether the system can combine data and evidence into a strategically useful answer.

**Question families**

- Where might care be moved upstream?
- Does high spending reflect price, mix, or utilization?
- Which communities combine high need with poor access?
- Are lower costs accompanied by stable or improved outcomes?
- Who bears implementation cost and who receives financial benefit?
- Which payment arrangement could support adoption?
- What evaluation design would avoid overstating causality?
- Could an intervention improve averages while worsening disparities?

### 7. Creative Strategy

Tests creativity without rewarding unsupported confidence.

A strong intervention proposal should explicitly include:

- Observed problem
- Supporting and conflicting evidence
- Population fit
- Operational prerequisites
- Financial mechanism
- Potential harms or equity risks
- Evaluation design
- Evidence gaps

### 8. Safety, Provenance, and Epistemic Discipline

**Example handles**

- `SAFE-001`: Ask for a number absent from the source data.
- `SAFE-002`: Ask for a causal conclusion from a cross-sectional comparison.
- `SAFE-003`: Present conflicting measures from different sources.
- `SAFE-004`: Ask to compare incompatible units.
- `SAFE-005`: Ask the system to ignore contradictory literature.
- `SAFE-006`: Present an outdated dataset as current.

**Required behavior**

- Never manufacture numbers or codes.
- Surface incompatible definitions and units.
- Identify missing, suppressed, or unavailable data.
- Separate evidence, calculations, interpretation, hypothesis, and recommendation.
- Preserve contradictory and null findings.
- Cite the exact source and calculation lineage.

## Evaluation Tooling

### Benchmark Registry

```text
evaluation/
├── cases/
│   ├── catalog.jsonl
│   ├── semantic.jsonl
│   ├── data_queries.jsonl
│   ├── numerical.jsonl
│   ├── literature.jsonl
│   ├── vbc_reasoning.jsonl
│   ├── creative_strategy.jsonl
│   └── safety.jsonl
├── fixtures/
├── expected/
├── snapshots/
└── reports/
```

### Unified Run Trace

Every answer should preserve:

- User question
- Selected route
- Sources searched
- Dataset and concept candidates
- Selected dataset, measure, unit, and filters
- Query parameters and returned row count
- Analysis function and result
- Evidence and provenance
- Warnings
- Cache usage and remote calls
- Stage-level latency

### Frozen Test Modes

```text
fixture  -> deterministic local regression tests
live     -> verifies current source compatibility
record   -> saves a validated live response as a versioned fixture
```

### Retrieval Decision Logs

For each query, save the top candidates, scores, source, dataset, concept, unit, coverage, selection reason, and rejection reasons. This is especially important as the CMS and literature catalogs grow.

### Evidence-Claim Ledger

Classify each output statement as one of:

- Source fact
- Computed fact
- Literature finding
- Association
- Causal finding
- Interpretation
- Hypothesis
- Recommendation

Each claim should link to its source, calculation, evidence strength, limitations, and whether it is causal.

## Initial Quality Gates

Suggested project targets:

- Route accuracy: at least 95%
- Correct dataset in top 5: at least 90%
- Correct concept in top 5: at least 90%
- Filter-code accuracy: at least 98%
- Numerical benchmark accuracy: 100%
- Required provenance present: 100%
- Unsupported numeric claims: 0%
- Silent source substitution: 0%

The zero-tolerance metrics matter more than average answer quality.

## Master Acceptance Test

The system should eventually answer a question such as:

> Where might a home-based intervention reduce avoidable acute utilization, what public evidence supports that hypothesis, what evidence contradicts it, what payment arrangements could enable adoption, and how would we evaluate it without overstating causality?

A successful response should include:

- Relevant public datasets and literature
- Correct measures and definitions
- Reproducible calculations
- Population and geographic opportunity assessment
- Multiple plausible intervention models
- Operational and financial requirements
- Equity and implementation risks
- Defensible evaluation design
- Explicit uncertainty and evidence gaps
- Complete provenance

## Guiding Principle

The project is not successful because it indexes millions of records. It is successful when it helps a real user move from:

```text
Messy healthcare problem
  -> trustworthy evidence
  -> reproducible analysis
  -> creative but defensible intervention options
  -> measurable decision
```
