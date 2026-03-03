# Repository Guidelines

## Project Structure & Module Organization
- `src/lib.rs` is the crate surface and re-exports the public API.
- Core OCR pipeline logic is split by stage: `src/det`, `src/cls`, `src/rec`, plus orchestration in `src/pipeline`.
- Runtime/provider wiring and session setup live in `src/runtime`; image operations live in `src/vision`.
- Input loading is in `src/input`, and output serializers/visualizers are in `src/output`.
- CLI binaries are in `src/bin`: `rapidocr.rs` for user-facing OCR commands and `bench_warm_e2e.rs` for performance runs.
- `assets/default_models.yaml` holds default model metadata. Local work directories like `target/`, `models/`, `test/`, and `test_output/` are ignored and should stay out of commits.

## Build, Test, and Development Commands
- `cargo check` performs fast compile-time validation while iterating.
- `cargo build --bin rapidocr` builds the main CLI with default features.
- `cargo run --bin rapidocr -- run --img-path <path-or-url> --json` runs OCR and prints JSON output.
- `cargo run --bin rapidocr -- config init --output ./default_rapidocr.yaml` writes a starter YAML config.
- `cargo test --lib --bins` runs unit tests in library and binary targets.
- `cargo run --bin bench_warm_e2e -- --images-dir test/test_files --rounds 3 --output test_output/bench.json` generates warm-run timing stats.
- Optional backend comparison builds: `cargo build --bin rapidocr --target-dir target/pure` and `cargo build --bin rapidocr --features opencv-backend --target-dir target/opencv`.

## Coding Style & Naming Conventions
- Use Rust 2024 idioms and run `cargo fmt --all` before committing.
- Keep lint clean with `cargo clippy --all-targets --all-features -- -D warnings`.
- Follow existing naming patterns: `snake_case` for modules/functions, `UpperCamelCase` for types/traits, `SCREAMING_SNAKE_CASE` for constants.
- In library code, prefer explicit `Result` propagation over `unwrap()`.

## Testing Guidelines
- Add unit tests next to implementation using `#[cfg(test)]` and `#[test]`.
- Name tests descriptively in `snake_case` (for example, `parse_rapidocr_yaml_directml_provider`).
- For OCR behavior changes, validate with CLI runs on representative images and compare JSON/Markdown outputs.

## Commit & Pull Request Guidelines
- Keep using Conventional Commit prefixes seen in history (for example, `feat: ...`, `refactor: ...`).
- Keep commits focused by concern; do not mix large refactors with behavior changes.
- PRs should include: what changed, why it changed, commands executed (`fmt`, `clippy`, `test`), and sample CLI output for user-visible OCR changes.
