# rembg — project tasks (run on the build host)
CARGO ?= cargo
FEATURES ?=
CARGO_FLAGS := $(if $(FEATURES),--features $(FEATURES),)
BIN := target/release/rembg

.PHONY: help build release cuda run gui test lint fmt check clean models dist

help: ## Show targets
	@grep -hE '^[a-z-]+:.*##' Makefile | awk -F':.*## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Debug build
	$(CARGO) build $(CARGO_FLAGS)

release: ## Release build (CPU; FEATURES=cuda for GPU)
	$(CARGO) build --release $(CARGO_FLAGS)

cuda: ## Release build with CUDA EP + bundle CUDA libs into target/release/lib
	$(CARGO) build --release --features cuda
	scripts/fetch-cuda-libs.sh target/release/lib

run: ## Run CLI: make run ARGS="remove in.jpg"
	$(CARGO) run --release $(CARGO_FLAGS) -- $(ARGS)

gui: ## Launch GUI
	$(CARGO) run --release $(CARGO_FLAGS) -- gui

test: ## Run tests
	$(CARGO) test --workspace $(CARGO_FLAGS)

lint: ## Clippy (deny warnings)
	$(CARGO) clippy --workspace --all-targets $(CARGO_FLAGS) -- -D warnings

fmt: ## Format code
	$(CARGO) fmt --all

check: ## fmt check + clippy + tests
	$(CARGO) fmt --all -- --check
	$(MAKE) lint test

models: release ## List models and install state
	$(BIN) models list

clean: ## cargo clean
	$(CARGO) clean
