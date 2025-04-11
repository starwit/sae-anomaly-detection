# Object-Detector model deployment container
This container copies all its model weights into a certain directory upon start.
The idea is to ship weights in images of this and use this as an init container in Kubernetes,
mounting a volume (as a weights target) that will then be mounted into the object-detector container
once init has completed.

## How to build
Move weight files into `weights` then run `make build push TAG=image_tag` (for tag naming see below).
The necessary weight files are as follows (names can be changed in settings, see `settings.template.yaml`):
- `model_weight.pth`
- `parameters.json`
- `frame.pth`

## Usage
Set up the container as an init container and mount the target volume at `/weights_target`.
### Kubernetes
Set this up as an init container.
### Docker Compose
Use `depends_on` with `condition: service_completed_successfully`.

## Tag format
`<model_type>-<camera>-<timestamp>`, e.g. `cnnasymmetricprob-rangelinesmedicaldr-202504071517` \
(all lowercase / numeric, i.e. `[a-z]+-[a-z]+-[0-9]+`)