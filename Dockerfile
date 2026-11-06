FROM starwitorg/base-python-image:0.0.15 AS build

# Copy only files that are necessary to install dependencies
COPY poetry.lock poetry.toml pyproject.toml /code/
WORKDIR /code
RUN poetry install

# Pre-download pretrained backbone weights (used for training), so no internet access is needed at runtime
ENV TORCH_HOME=/code/.torch
RUN .venv/bin/python -c "from torchvision.models import MobileNet_V3_Small_Weights; MobileNet_V3_Small_Weights.IMAGENET1K_V1.get_state_dict()"

# Copy the rest of the project
COPY . /code/

### Main artifact / deliverable image

FROM python:3.12-slim
RUN apt update && apt install --no-install-recommends -y \
    libglib2.0-0 \
    libgl1 \
    libturbojpeg0

# Create a non-root user and group
RUN addgroup --system --gid 1000 appgroup && adduser --system --uid 1000 --ingroup appgroup --home /home/appuser appuser

COPY --from=build --chown=appuser:appgroup /code /code
WORKDIR /code

# Mount point for training data and model (e.g. PVC)
RUN mkdir -p /code/model && chown appuser:appgroup /code/model

# Switch to non-root user
USER appuser

ENV PATH="/code/.venv/bin:$PATH"
ENV TORCH_HOME=/code/.torch
CMD [ "python", "main.py" ]