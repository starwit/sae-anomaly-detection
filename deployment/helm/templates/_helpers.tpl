{{/*
Expand the name of the chart.
*/}}
{{- define "sae-anomaly-detection.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "sae-anomaly-detection.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "sae-anomaly-detection.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "sae-anomaly-detection.labels" -}}
helm.sh/chart: {{ include "sae-anomaly-detection.chart" . }}
{{ include "sae-anomaly-detection.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "sae-anomaly-detection.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sae-anomaly-detection.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "sae-anomaly-detection.serviceAccountName" -}}
{{- include "sae-anomaly-detection.fullname" . }}
{{- end }}

{{/*
Name of the PVC holding training data and model (either existing or chart-managed)
*/}}
{{- define "sae-anomaly-detection.pvcName" -}}
{{- .Values.persistence.existingClaim | default (printf "%s-data" (include "sae-anomaly-detection.fullname" .)) }}
{{- end }}
