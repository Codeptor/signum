# Style and Conventions

## Python
- Use type hints throughout
- Follow PEP 8 naming (snake_case for functions/variables, PascalCase for classes)
- Use `ruff` for linting and formatting
- Docstrings only where logic isn't self-evident
- Managed via `pyproject.toml`

## Rust
- Follow standard Rust conventions (snake_case, PascalCase for types)
- Use `clippy` for linting
- Cargo workspace at project root
- Benchmark with Criterion.rs

## General
- Monorepo structure
- YAGNI — minimal complexity for current requirements
- Tests alongside implementation
- DVC for data, Git for code
