IMAGE_NAME := mcs-pcr
CONTAINER_NAME := mcs-pcr-run
DATA_DIR := $(shell pwd)/data

.PHONY: build run shell clean rebuild

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run registration (pass args via ARGS, mount dataset via DATA_DIR)
# Example: make run ARGS="Apartment 0 line_20 20 3.0 2.0" DATA_DIR=/path/to/dataset
run:
	docker run --rm \
		--name $(CONTAINER_NAME) \
		-v $(DATA_DIR):/data \
		-v $(shell pwd)/configs:/app/configs \
		-v $(shell pwd)/reg_results:/app/reg_results \
		$(IMAGE_NAME) $(ARGS)

# Open an interactive shell in the container
shell:
	docker run --rm -it \
		-v $(DATA_DIR):/data \
		-v $(shell pwd)/configs:/app/configs \
		-v $(shell pwd)/reg_results:/app/reg_results \
		$(IMAGE_NAME) /bin/bash

# Remove the Docker image
clean:
	docker rmi $(IMAGE_NAME) 2>/dev/null || true

# Full rebuild (no cache)
rebuild:
	docker build --no-cache -t $(IMAGE_NAME) .
