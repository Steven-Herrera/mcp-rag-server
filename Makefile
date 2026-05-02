.PHONY: help install lint fmt fmt-check test build docker-build docker-push docker-lint ci clean

UV              := uv
REGISTRY        ?= ghcr.io/steven-herrera
IMAGE_NAME      ?= mcp-rag-server
VERSION         ?= v0.1.11
IMAGE           := $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
DOCKERFILE      ?= Dockerfile
DOCKER_CONTEXT  ?= .
HADOLINT_IMAGE  ?= hadolint/hadolint:latest-debian

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:
	$(UV) sync --all-extras

fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

fmt-check:
	$(UV) run ruff format --check .

lint:
	$(UV) run ruff check .
	$(UV) run pylint mcp_server/

test:
	$(UV) run pytest --cov=mcp_server --cov-report=term-missing -q

build:
	$(UV) build

docker-build:
	docker build -f $(DOCKERFILE) -t $(IMAGE) $(DOCKER_CONTEXT)
	@echo "Built: $(IMAGE)"

docker-push:
	docker push $(IMAGE)
	@echo "Pushed: $(IMAGE)"

docker-lint:
	docker run --rm -i $(HADOLINT_IMAGE) < $(DOCKERFILE)

ci: install fmt-check lint test docker-lint

clean: ## Remove caches and build artifacts
	rm -rf .ruff_cache .pytest_cache .mypy_cache dist __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true