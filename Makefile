# rembg — project tasks (run on the build host)
CARGO ?= cargo
FEATURES ?=
CARGO_FLAGS := $(if $(FEATURES),--features $(addprefix rembg-cli/,$(FEATURES)),)
BIN := target/release/rembg

.PHONY: help build release cuda cuda-libs run gui test lint fmt check clean models bench screenshots

help: ## Show targets
	@grep -hE '^[a-z-]+:.*##' Makefile | awk -F':.*## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Debug build
	$(CARGO) build $(CARGO_FLAGS)

release: ## Release build (CPU; FEATURES=cuda for GPU)
	$(CARGO) build --release $(CARGO_FLAGS)

cuda: ## Release build with CUDA EP + bundle CUDA 13 runtime libs into target/release/lib
	$(CARGO) build --release --features rembg-cli/cuda
	scripts/fetch-cuda-libs.sh target/release/lib

cuda-libs: ## (Re)fetch only the bundled CUDA runtime libs
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

bench: release ## Time every installed model on a sample image: make bench IMG=photo.jpg
	$(BIN) compare $(IMG) -o /tmp/rembg-bench.png --tile 256

screenshots: release ## Regenerate docs/screenshots (needs a display + models)
	scripts/screenshots.sh

clean: ## cargo clean
	$(CARGO) clean
