//! Binary entry point for the `castforge` CLI.

fn main() -> anyhow::Result<()> {
    castforge::cli::run()
}
