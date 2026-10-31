# Contributing to CastForge

Thank you for your interest in contributing to **CastForge**!

## Code Quality & CI Requirements

Every pull request is validated against strict repository checks:

1. **Formatting**: `cargo fmt --all -- --check` must pass with zero diffs.
2. **100-Column Line Limit**: Every tracked file must have a maximum line length of `<= 100`
   characters:
   ```bash
   git ls-files | xargs awk 'length($0) > 100 { print FILENAME ":" FNR; exit 1 }'
   ```
3. **Zero Clippy Warnings**: `cargo clippy --all-targets -- -D warnings` must pass cleanly.
4. **Zero Rustdoc Warnings**: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` must pass cleanly.
5. **Module Hygiene**: All `mod.rs` files must contain only module declarations (`pub mod ...;`)
   and re-exports (`pub use ...;`), with no inline implementation logic.
6. **Test Suite**: `RUSTFLAGS="-D warnings" cargo test` must pass across all unit, integration,
   fuzz/property, negative, and performance scaling suites.

## Adding a New Optimization Pass

1. Implement the [`Pass`](src/passes/registry/manager.rs) trait in the appropriate domain package
   under `src/passes/<package>/`.
2. Register the pass in [`config/passes.toml`](config/passes.toml) and wire its constructor in
   `src/passes/registry/leaf.rs` or `src/passes/registry/families.rs`.
3. Add numerical equivalence and round-trip tests in `tests/`.
