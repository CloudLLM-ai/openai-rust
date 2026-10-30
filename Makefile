.PHONY: fmt clippy test check build doc publish clean help

CARGO := /usr/bin/env cargo
# cargo-sweep (cargo install cargo-sweep), when installed, deletes whatever an older build
# left in target/ that this build didn't use, so only fresh release artifacts remain.
SWEEP := $(shell command -v cargo-sweep >/dev/null 2>&1 && echo cargo sweep)

default: help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

fmt: ## Format sources
	$(CARGO) fmt

clippy: fmt ## Clippy with -D warnings (fmt first)
	$(CARGO) clippy --release --all-targets --all-features -- -D warnings

check: ## Typecheck without building artifacts
	$(CARGO) check --release --all-targets --all-features

test: ## Run unit + integration tests (live tests skip without OPENAI_API_KEY)
	$(CARGO) test --release --all-features

build: fmt ## Release build, then sweep stale artifacts
	$(if $(SWEEP),$(SWEEP) --stamp)
	$(CARGO) build --release --all-features
	$(if $(SWEEP),$(SWEEP) --file)

doc: ## Generate rustdoc
	$(CARGO) doc --release --no-deps --all-features

publish: clippy test ## Publish to crates.io after clippy + tests
	$(CARGO) publish

clean: ## Remove target/
	$(CARGO) clean
