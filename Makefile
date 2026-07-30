# Short names for the long commands. Every target runs inside the Nix dev shell
# via scripts/x, so `make test` works identically for you, for CI, and for an
# agent in a worktree with no direnv loaded.
#
#   make            list targets
#   make test       whole workspace
#   make t-model    one lane, fast loop
#   make smoke-cuda the Phase 0 GPU gate
#
# Agents: use these. If you need a target that does not exist, request it —
# see ROADMAP §9. The Makefile is orchestrator-owned; you cannot edit it.

X := ./scripts/x
CARGO := $(X) cargo

.DEFAULT_GOAL := help

# ---------------------------------------------------------------- build & test

.PHONY: build
build: ## Build the whole workspace
	$(CARGO) build --workspace

.PHONY: test
test: ## Test the whole workspace
	$(CARGO) test --workspace

.PHONY: check
check: ## Fast type-check, no codegen
	$(CARGO) check --workspace --all-targets

.PHONY: fmt
fmt: ## Format
	$(CARGO) fmt --all

.PHONY: fmt-check
fmt-check: ## Verify formatting without writing
	$(CARGO) fmt --all -- --check

.PHONY: clippy
clippy: ## Lint; warnings are errors
	$(CARGO) clippy --workspace --all-targets -- -D warnings

.PHONY: gate
gate: fmt-check clippy test ## Everything a lane must pass before review
	@echo "gate: PASS"

# ------------------------------------------------------------------ lane loops
# One target per ownership lane (.jane/ownership), for a tight iteration loop
# that doesn't wait on unrelated crates.

.PHONY: t-model
t-model: ## Test the model-config lane
	$(CARGO) test -p jane-model

.PHONY: t-data
t-data: ## Test all of jane-data
	$(CARGO) test -p jane-data

.PHONY: t-tokenizer
t-tokenizer: ## Test the tokenizer lane (tokenizer + binarize)
	$(CARGO) test -p jane-data tokenizer:: binarize::

.PHONY: t-dataset
t-dataset: ## Test the dataset lane (dataset + batcher)
	$(CARGO) test -p jane-data dataset:: batcher::

.PHONY: t-sources
t-sources: ## Test the sources lane (corpus acquisition)
	$(CARGO) test -p jane-data source::

.PHONY: t-train
t-train: ## Test the train-config lane
	$(CARGO) test -p jane-train

.PHONY: t-ignored
t-ignored: ## Run the network/Python tests that are #[ignore]d by default
	$(CARGO) test --workspace -- --ignored

# ------------------------------------------------------------------- phase 0/1

.PHONY: smoke
smoke: ## Phase 0 gate on CPU
	$(CARGO) run -q -p jane-cli -- smoke --backend ndarray --size 512

.PHONY: smoke-cuda
smoke-cuda: ## Phase 0 gate on GPU — proves CubeCL compiles kernels for sm_120
	$(CARGO) run -q -p jane-cli --features cuda -- smoke --backend cuda --size 2048

.PHONY: prepare
prepare: ## Phase 1 pipeline on the rung-0 corpus (fast)
	$(CARGO) run -q -p jane-cli -- prepare --corpus tiny-shakespeare --vocab-size 4096

# --------------------------------------------------------------------- harness

.PHONY: hooks
hooks: ## Install the enforcement hooks (run once per clone)
	./scripts/setup-hooks.sh

.PHONY: status
status: ## Which lanes are in flight, reconciled against live git state
	./scripts/flight.sh

.PHONY: wt
wt: ## List agent worktrees
	./scripts/agent-worktree.sh list

.PHONY: sccache-stats
sccache-stats: ## Cache hit rate — tells you if the harness is actually sharing work
	$(X) sccache --show-stats

.PHONY: clean
clean: ## Remove build artifacts (keeps the sccache and data/)
	$(CARGO) clean

.PHONY: help
help: ## List targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
