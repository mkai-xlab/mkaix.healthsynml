.DEFAULT_GOAL := help

AI_IMAGE := knee-oa-ai:local
AI_CONTAINER := knee-oa-ai
AI_PORT ?= 8005

.PHONY: help ai-build ai-up ai-down ai-logs ai-health up down status test experiments

help:
	@printf '%s\n' 'make ai-up       Build and start the API and embedded result viewer on port $(AI_PORT)' 'make up          Start the API' 'make down        Stop the API container' 'make status      Show service state' 'make test        Run pytest in the API image' 'make experiments Regenerate experiment summaries'

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

up: ai-up

down: ai-down

status:
	docker ps --filter name=$(AI_CONTAINER)

test: ai-build
	docker run --rm -v $(CURDIR)/tests:/app/tests:ro $(AI_IMAGE) python -m pytest -q

experiments:
	python3 scripts/build_experiment_summary.py
