.DEFAULT_GOAL := help

AI_IMAGE := knee-oa-ai:local
AI_CONTAINER := knee-oa-ai
VIEWER_IMAGE := kl-response-viewer:latest
VIEWER_CONTAINER := kl-response-viewer
AI_PORT ?= 8005
VIEWER_PORT ?= 8088

.PHONY: help ai-build ai-up ai-down ai-logs ai-health viewer-build viewer-up viewer-down viewer-logs up down status test experiments

help:
	@printf '%s\n' 'make ai-up       Build and start the DenseNet Grad-CAM API on port $(AI_PORT)' 'make viewer-up   Build and start the response viewer on port $(VIEWER_PORT)' 'make up          Start API and response viewer' 'make down        Stop both local containers' 'make status      Show service state' 'make test        Run pytest in the API image' 'make experiments Regenerate docs/report/all_experiments.xlsx and CSV tabs'

ai-build:
	docker build -t $(AI_IMAGE) .

ai-up: ai-build
	-docker rm -f $(AI_CONTAINER)
	docker run -d --name $(AI_CONTAINER) --env-file ../env/ai.env -p $(AI_PORT):8005 -v $(CURDIR)/checkpoints:/app/checkpoints:ro $(AI_IMAGE)

ai-down:
	-docker rm -f $(AI_CONTAINER)

ai-logs:
	docker logs -f $(AI_CONTAINER)

ai-health:
	curl --fail --silent --show-error http://127.0.0.1:$(AI_PORT)/api/v1/health

viewer-build:
	docker build -t $(VIEWER_IMAGE) tools/kl_response_viewer

viewer-up: viewer-build
	-docker rm -f $(VIEWER_CONTAINER)
	docker run -d --name $(VIEWER_CONTAINER) -p $(VIEWER_PORT):8080 $(VIEWER_IMAGE)

viewer-down:
	-docker rm -f $(VIEWER_CONTAINER)

viewer-logs:
	docker logs -f $(VIEWER_CONTAINER)

up: ai-up viewer-up

down: ai-down viewer-down

status:
	docker ps --filter name=$(AI_CONTAINER) --filter name=$(VIEWER_CONTAINER)

test: ai-build
	docker run --rm -v $(CURDIR)/tests:/app/tests:ro $(AI_IMAGE) python -m pytest -q

experiments:
	python3 scripts/build_experiment_summary.py
