.PHONY: build push check-tag

# Check if TAG environment variable is set
check-tag:
	@if [ -z "$(TAG)" ]; then \
		echo "Error: TAG environment variable is not set"; \
		echo "Usage: make build / push TAG=your_tag_name"; \
		exit 1; \
	fi

# Build the Docker image
build: check-tag
	docker build -t docker.internal.starwit-infra.de/sae-anomaly-detection-weights:$(TAG) .

# Push the Docker image
push: check-tag
	docker push docker.internal.starwit-infra.de/sae-anomaly-detection-weights:$(TAG)
