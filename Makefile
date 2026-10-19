# Each package under packages/ is an independent uv project: its own
# pyproject.toml, uv.lock and .venv, resolved on its own. There is no uv
# workspace, which would force one shared environment -- and one shared
# environment is exactly what this repo cannot have, because target_server's
# vLLM needs transformers >= 5.10.4 while tinker-cookbook declares <= 5.5.4.
# See the dependency note in README.md.
#
# Don't run `uv run pytest` from the repo root: there is no root project.

PKGS := common investigator_trainer cbrn_eval sft_investigator sft_lora_pipeline
TEST_PKGS := common investigator_trainer
# Linux + CUDA only, and deliberately outside PKGS: syncing it on a laptop
# tries to build vLLM.
GPU_PKGS := target_server sft_lora_pipeline

.PHONY: help lock sync test lint format sync-server
help:
	@echo "  make sync         - uv sync every CPU package ($(words $(PKGS)) of them)"
	@echo "  make test         - pytest in $(TEST_PKGS)"
	@echo "  make lock         - uv lock every package, including the GPU ones"
	@echo "  make lint/format  - ruff across the repo"
	@echo "  make sync-server  - uv sync target_server (Linux + CUDA only)"

lock:
	@$(foreach p,$(PKGS) $(GPU_PKGS),echo ">> lock $(p)"; (cd packages/$(p) && uv lock) || exit 1;)

sync:
	@$(foreach p,$(PKGS),echo ">> sync $(p)"; (cd packages/$(p) && uv sync) || exit 1;)

sync-server:
	cd packages/target_server && uv sync

test:
	@$(foreach p,$(TEST_PKGS),echo ">> pytest $(p)"; (cd packages/$(p) && uv run pytest -q) || exit 1;)

lint:
	uvx ruff@0.15.6 check packages

format:
	uvx ruff@0.15.6 format packages
