.PHONY: fmt clippy test check build doc publish clean help

CARGO := /usr/bin/env cargo

default: help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

fmt: ## Format sources
	$(CARGO) fmt

clippy: fmt ## Clippy with -D warnings (fmt first)
	$(CARGO) clippy --all-targets --all-features -- -D warnings

check: ## Typecheck without building artifacts
	$(CARGO) check --all-targets --all-features

test: ## Run unit + integration tests (live tests skip without OPENAI_API_KEY)
	$(CARGO) test --all-features

build: fmt ## Debug build
	$(CARGO) build --all-features

doc: ## Generate rustdoc
	$(CARGO) doc --no-deps --all-features

publish: clippy test ## Publish to crates.io after clippy + tests
	$(CARGO) publish

clean: ## Remove target/
	$(CARGO) clean
