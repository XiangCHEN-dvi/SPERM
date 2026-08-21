# Architecture

SPERM uses a `src` layout and separates the shared prior language from each
model family's enforcement algorithm.

```text
sperm/
├── pyproject.toml                 # Package metadata and tool configuration
├── src/
│   └── sperm/
│       ├── __init__.py            # Package version and top-level metadata
│       ├── priors/
│       │   ├── _priors.py         # Canonical, immutable prior primitives
│       │   ├── _parser.py         # Public input parsing and normalization
│       │   └── _compiler.py       # Validation and combination rules
│       ├── linear_model/           # Coefficient-constrained linear models
│       ├── tree_model/             # Constrained tree-growing algorithms
│       ├── neural_network/         # MLP and input-convex architectures
│       └── gaussian_process/       # Spline posterior and constraints
├── tests/                          # Unit, regression, and estimator checks
├── examples/                       # Executable plotting experiments
└── docs/                           # MyST/Sphinx documentation source
```

## Data Flow

```text
Public prior declaration
        │
        ▼
Parse and canonicalize (`sperm.priors`)
        │
        ▼
Validate and simplify combinations
        │
        ▼
Compile into a model-native mechanism
        │
        ├── Linear ── coefficient constraints
        ├── Tree ──── constrained split and leaf decisions
        ├── MLP ───── shape-preserving parameterization
        └── GPR ───── spline-coefficient constraints
        │
        ▼
scikit-learn-compatible estimator (`fit` / `predict`)
```

A prior's mathematical meaning belongs in `sperm.priors`; enforcement belongs
in the relevant estimator subpackage. This keeps the vocabulary consistent
without forcing unrelated model families through one algorithm.
