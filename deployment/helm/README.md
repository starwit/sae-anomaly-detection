# SAE Anomaly Detection Helm Chart
Deploys `sae-anomaly-detection` next to an existing [SAE](https://github.com/starwit/starwit-awareness-engine) installation (it reads from and writes to the SAE Valkey).

# How-To Install
- Copy `customvalues.template.yaml` to `customvalues.yaml` and adapt it (esp. Valkey host and stream settings)
- `helm install sae-anomaly-detection oci://registry-1.docker.io/starwitorg/sae-anomaly-detection-chart --version <version> -f customvalues.yaml`

# Training and Persistence
Training data and the model live on a PVC mounted at `/code/model` (see `persistence`). The chart-managed PVC is annotated with `helm.sh/resource-policy: keep`, i.e. it is not deleted on `helm uninstall`.
- To train a model, set `settingsYaml.model.start_training: true`. The component then collects data for `time_interval_train_data` hours (no detection during that time), trains a model and switches over to detection.
- Training data is buffered in memory and only written to the volume when training starts. If the pod restarts during the collection window, the collected data is lost and collection starts over.
- Each training run stores its data in a new folder `<path_train_data>_<timestamp>` on the volume. Old folders are not cleaned up automatically.
- If a model (`parameters_path` and `weights_path`) already exists, training is skipped, so pod restarts do not trigger retraining. This also applies to `helm uninstall` and a fresh `helm install`, as the PVC (and thereby the model) is kept.
- To retrain, delete the model files and restart the pod (training data folders are kept):
  ```sh
  kubectl exec <pod> -- rm /code/model/parameters.json /code/model/model_weights.pth
  kubectl delete pod <pod>
  ```
- To start from scratch (removes model and all training data), uninstall the release and delete the PVC (named `<fullname>-data`, unless `persistence.existingClaim` is set):
  ```sh
  helm uninstall <release>
  kubectl delete pvc <fullname>-data
  ```
- With `persistence.enabled: false` an `emptyDir` is used instead, i.e. the model is lost whenever the pod is recreated (it is retrained if `start_training` is true; otherwise the component fails to start without a model).

# Changelog
**Breaking changes (esp. with regard to config format) should only happen on major version bumps**

## 2.1.0
- Chart and image are publicly available, no pull secret needed (`oci://registry-1.docker.io/starwitorg/sae-anomaly-detection-chart`)

## 2.0.2
- More readable logs, incl. details on detected anomalies
- Works with input messages without frame data (grey placeholder image)

## 2.0.1
- Replaying dumps now works correctly (frame timestamps instead of wall-clock time)
- Waits for the output Valkey to be reachable before starting (additional init container)

## 2.0.0
- Models can be trained directly in the component, no weights image needed anymore (`settingsYaml.model.start_training`)
- Training data and model survive restarts and uninstalls (PVC, see `persistence`)
- Breaking: config moved from `config` to `settingsYaml`
- Breaking: pull secrets have to be provided externally (`extraImagePullSecrets`)
- Breaking: resource names are derived from the release name
- Works out of the box next to an SAE release named `sae` (Valkey host `sae-valkey`)
- Resources, security contexts and scheduling are configurable
