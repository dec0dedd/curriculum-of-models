# Each package under packages/ is an independent uv project: its own
# pyproject.toml, uv.lock and .venv, resolved on its own. There is no uv
# workspace, which would force one shared environment -- and one shared
# environment is exactly what this repo cannot have, because target_server's
# vLLM needs transformers >= 5.10.4 while tinker-cookbook declares <= 5.5.4.
# See the dependency note in README.md.
#
# Don't run `uv run pytest` from the repo root: there is no root project.

PKGS := common investigator_trainer cbrn_eval target_server

.PHONY: help lock sync test lint format sync-server
help:
	@echo "  make sync         - uv sync every package ($(words $(PKGS)) of them)"
	@echo "  make test         - pytest in every package"
	@echo "  make lock         - uv lock every package"
	@echo "  make lint/format  - ruff across the repo"
	@echo "  make sync-server  - uv sync target_server (Linux + CUDA only)"

lock:
	@$(foreach p,$(PKGS),echo ">> lock $(p)"; (cd packages/$(p) && uv lock) || exit 1;)

sync:
	@$(foreach p,$(PKGS),echo ">> sync $(p)"; (cd packages/$(p) && uv sync) || exit 1;)

sync-server:
	cd packages/target_server && uv sync --group vllm

test:
	@$(foreach p,$(PKGS),echo ">> pytest $(p)"; (cd packages/$(p) && uv run pytest -q) || exit 1;)

lint:
	uvx ruff@0.15.6 check packages

format:
	uvx ruff@0.15.6 format packages
