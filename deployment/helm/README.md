# SAE Anomaly Detection Helm Chart
Deploys `sae-anomaly-detection` next to an existing [SAE](https://github.com/starwit/starwit-awareness-engine) installation (it reads from and writes to the SAE Valkey).

# How-To Install
- Images are hosted in the internal registry, i.e. create a pull secret first and reference it in `extraImagePullSecrets`:
  `kubectl create secret docker-registry starwit-internal-registry --docker-server=docker.internal.starwit-infra.de --docker-username=<user> --docker-password=<password>`
- Copy `customvalues.template.yaml` to `customvalues.yaml` and adapt it (esp. Valkey host and stream settings)
- `helm install sae-anomaly-detection <chart> -f customvalues.yaml`

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
## Next major version
- Align chart with SAE helm chart conventions
  - Breaking config change: `config` is now `settingsYaml`
  - Breaking config change: the chart does not create a pull secret anymore (`existingPullSecret` and `pullSecret` are removed); use `extraImagePullSecrets` instead
  - Breaking config change: the weights init container is removed (`weights` is removed); models are trained into the persistent volume
  - Default Valkey host is now `sae-valkey` (Valkey service of an SAE release named `sae`)
  - Pod annotations are now `sae.starwit.de/*`; a `wait-for-valkey` init container was added
  - `resources`, `podSecurityContext` (default `fsGroup: 1000`), `securityContext`, `nodeSelector`, `affinity` and `tolerations` are configurable
- Add PVC for training data and model (`persistence`)
