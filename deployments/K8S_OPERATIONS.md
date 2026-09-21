# Project Synapse: Enterprise Kubernetes Operational Runbook

**Production Infrastructure, GitOps Lifecycle & SRE Handbook**  
*Target Release:* `v1.0.0` | *API Version:* `autoscaling/v2`, `apps/v1`, `helm.sh/v3`

---

## Table of Contents
1. [Architectural Overview](#1-architectural-overview)
2. [Cluster Provisioning & GPU Acceleration](#2-cluster-provisioning--gpu-acceleration)
   - [Google Kubernetes Engine (GKE)](#21-google-kubernetes-engine-gke)
   - [AWS Elastic Kubernetes Service (EKS)](#22-aws-elastic-kubernetes-service-eks)
   - [Bare-Metal Kubernetes with NVIDIA GPU Operator](#23-bare-metal-kubernetes-with-nvidia-gpu-operator)
3. [Zero-Key Secret Management via Cloud Workload Identity](#3-zero-key-secret-management-via-cloud-workload-identity)
4. [Helm Chart Packaging & Lifecycle Management](#4-helm-chart-packaging--lifecycle-management)
   - [Inspection & Value Configuration](#41-inspection--value-configuration)
   - [Dry-Run Template Rendering](#42-dry-run-template-rendering)
   - [Deployment & Atomic Upgrades](#43-deployment--atomic-upgrades)
5. [Horizontal Pod Autoscaling (HPA v2) & Metrics](#5-horizontal-pod-autoscaling-hpa-v2--metrics)
6. [Rollback Strategies, Chaos Survivability & Disaster Recovery](#6-rollback-strategies-chaos-survivability--disaster-recovery)
7. [Observability & Prometheus Monitoring](#7-observability--prometheus-monitoring)

---

## 1. Architectural Overview

Project Synapse is an enterprise autonomous agentic cognitive architecture operating across two decoupled Kubernetes microservices:

```mermaid
graph TD
    Client([External Client / Agent Gateway]) -->|HTTPS /v1/chat/completions| Ingress[NGINX Ingress Controller]
    Ingress --> Service[ClusterIP Service: agi-synapse:8000]
    
    subgraph Serving Pods [agi-serving Autoscaled Deployment]
        Pod1[agi-serving-pod-1<br/>T4 GPU / PEFT LoRA]
        Pod2[agi-serving-pod-2<br/>T4 GPU / PEFT LoRA]
        PodN[agi-serving-pod-N<br/>Autoscaled via HPA v2]
    end
    Service --> Pod1
    Service --> Pod2
    Service --> PodN

    subgraph Memory & Cloud Persistence
        GCS[(Google Cloud Storage<br/>5TB gs://agi-agent-ingestion-data/)]
        MySQL[(MySQL 8.0<br/>Relational Papers & Trajectories)]
        PVC[(Persistent Volume Claim<br/>20Gi ReadWriteOnce)]
    end

    subgraph Daemon Pod [agi-daemon Deployment]
        Daemon[agi-daemon<br/>Continuous Ingestion & FAISS Sync]
    end

    Pod1 -.->|Workload Identity| GCS
    Pod2 -.->|Workload Identity| GCS
    Daemon -.->|Workload Identity| GCS
    Daemon -->|Index Flush| PVC
    Daemon -->|Metadata Storage| MySQL
    Pod1 -->|Audit Logging| MySQL
    
    HPA[Horizontal Pod Autoscaler v2<br/>CPU > 75% | Latency > 250ms | Concurrency > 15] -.->|Scales| Serving Pods
```

---

## 2. Cluster Provisioning & GPU Acceleration

### 2.1. Google Kubernetes Engine (GKE)

Provision a regional GKE cluster with Workload Identity, GPU acceleration, and preemptible/spot node pools:

```bash
# 1. Create Regional Cluster with GKE Workload Identity enabled
gcloud container clusters create synapse-prod-cluster \
    --region us-central1 \
    --release-channel regular \
    --workload-pool=agi-cloud-prod.svc.id.goog \
    --enable-autoscaling \
    --min-nodes=2 \
    --max-nodes=10 \
    --num-nodes=3 \
    --machine-type=e2-standard-4 \
    --enable-ip-alias

# 2. Add GPU-Accelerated Node Pool (NVIDIA Tesla T4 with Auto-Installation)
gcloud container node-pools create gpu-inference-pool \
    --cluster=synapse-prod-cluster \
    --region=us-central1 \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1,gpu-driver-version=default \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=8 \
    --num-nodes=2 \
    --node-taints=nvidia.com/gpu=present:NoSchedule

# 3. Obtain Kubeconfig Credentials
gcloud container clusters get-credentials synapse-prod-cluster --region us-central1
```

### 2.2. AWS Elastic Kubernetes Service (EKS)

For AWS deployments with NVIDIA A10G/T4 GPUs (`g4dn.xlarge` or `g5.xlarge`):

```bash
eksctl create cluster \
  --name synapse-eks-prod \
  --region us-west-2 \
  --nodegroup-name gpu-inference-workers \
  --node-type g4dn.xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 8 \
  --enable-auto-mode \
  --with-oidc
```

### 2.3. Bare-Metal Kubernetes with NVIDIA GPU Operator

On private on-premise clusters or edge Kubernetes instances:

```bash
# Install NVIDIA Container Toolkit and GPU Operator via Helm
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator \
     --set driver.enabled=true \
     --set toolkit.enabled=true
```

---

## 3. Zero-Key Secret Management via Cloud Workload Identity

Static JSON service account keys represent severe security liabilities in cloud-native production. Project Synapse binds Kubernetes Service Accounts directly to Google Cloud IAM roles via **Workload Identity**.

### 3.1. Bind IAM Role to Kubernetes ServiceAccount

```bash
# Set environment variables
export PROJECT_ID="agi-cloud-prod"
export K8S_NAMESPACE="production"
export K8S_SA_NAME="agi-synapse-sa"
export GCP_SA_EMAIL="agi-agent-workload-identity@${PROJECT_ID}.iam.gserviceaccount.com"

# 1. Create the GCP Service Account
gcloud iam service-accounts create agi-agent-workload-identity \
    --display-name="Synapse GCS & Ingestion Agent Service Account" \
    --project=${PROJECT_ID}

# 2. Grant Storage Admin and BigQuery/Metrics permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${GCP_SA_EMAIL}" \
    --role="roles/storage.objectAdmin"

# 3. Allow K8s ServiceAccount to impersonate the GCP Service Account
gcloud iam service-accounts add-iam-policy-binding ${GCP_SA_EMAIL} \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${K8S_NAMESPACE}/${K8S_SA_NAME}]" \
    --project=${PROJECT_ID}
```

In `values.yaml`, ensure Workload Identity annotation is injected:

```yaml
serviceAccount:
  create: true
  name: "agi-synapse-sa"
  annotations:
    iam.gke.io/gcp-service-account: "agi-agent-workload-identity@agi-cloud-prod.iam.gserviceaccount.com"
```

---

## 4. Helm Chart Packaging & Lifecycle Management

The Helm chart is located at `deployments/helm/agi-synapse/`.

### 4.1. Inspection & Value Configuration

Verify chart structure and syntax:

```bash
helm lint ./deployments/helm/agi-synapse --strict
```

### 4.2. Dry-Run Template Rendering

Validate complete manifest rendering without touching the cluster:

```bash
helm template agi-synapse ./deployments/helm/agi-synapse \
    --namespace production \
    --values ./deployments/helm/agi-synapse/values.yaml \
    --set global.environment=production > /tmp/rendered_manifest.yaml

# Verify resource specifications
kubectl apply --dry-run=client -f /tmp/rendered_manifest.yaml
```

### 4.3. Deployment & Atomic Upgrades

Deploy Project Synapse into the `production` namespace with atomic rollback protection:

```bash
# Create namespace
kubectl create namespace production --dry-run=client -o yaml | kubectl apply -f -

# Install or upgrade release atomically
helm upgrade --install agi-synapse ./deployments/helm/agi-synapse \
    --namespace production \
    --values ./deployments/helm/agi-synapse/values.yaml \
    --atomic \
    --timeout 10m \
    --create-namespace
```

### 4.4. Zero-Downtime Rolling Update Strategy

The serving deployment enforces:
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 0
```
This ensures new GPU inference pods must pass their `/health` readiness probes before older pods receive SIGTERM, guaranteeing zero dropped HTTP connections.

---

## 5. Horizontal Pod Autoscaling (HPA v2) & Metrics

Autoscaling is governed by `autoscaling/v2` with multi-metric evaluations:

```yaml
spec:
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 75
    - type: Pods
      pods:
        metric:
          name: agi_serving_ttft_seconds
        target:
          type: AverageValue
          averageValue: "250m"
    - type: Pods
      pods:
        metric:
          name: agi_serving_active_requests
        target:
          type: AverageValue
          averageValue: 15
```

### 5.1. Inspecting Live Autoscaling State

```bash
# Check HPA status and current metric consumption
kubectl get hpa agi-synapse-serving-hpa -n production -w

# Describe detailed scaling events and metric calculations
kubectl describe hpa agi-synapse-serving-hpa -n production
```

---

## 6. Rollback Strategies, Chaos Survivability & Disaster Recovery

### 6.1. Immediate Release Rollback

If a newly deployed LoRA adapter or image tag introduces regressions:

```bash
# Check release revision history
helm history agi-synapse -n production

# Rollback to the previous stable revision instantly
helm rollback agi-synapse 0 -n production --wait
```

### 6.2. Pod Disruption Budgets (PDB)

To prevent voluntary disruptions (cluster upgrades, node draining) from dropping serving capacity:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: agi-serving-pdb
  namespace: production
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: agi-synapse
      app.kubernetes.io/component: serving
```

### 6.3. Preemptible Node & Spot Eviction Survivability

When GKE or AWS preemption occurs:
1. Kubernetes sends `SIGTERM` to the container.
2. The FastAPI `lifespan` handler receives the signal and stops accepting new requests while completing active inferences within a 30-second graceful termination window (`terminationGracePeriodSeconds: 30`).
3. Readiness probe fails immediately, removing the pod from Ingress endpoints.
4. Kubernetes cluster autoscaler provisions a replacement node on demand.

### 6.4. FAISS Persistent Index Recovery

If the `agi-daemon` pod is rescheduled:
- The persistent volume (`data_memory`) re-attaches automatically.
- The daemon checks `index.faiss` and `metadata.json`. If corrupt, it downloads the authoritative snapshot directly from `gs://agi-agent-ingestion-data/models/vector_memory/`.

---

## 7. Observability & Prometheus Monitoring

### 7.1. Prometheus Operator Scrape Endpoints

The Helm chart packages a `ServiceMonitor` resource:
- Serving Microservice: Port `8000`, Path `/metrics` (TTFT, TPS, inference latency percentiles, VRAM allocated).
- Ingestion Daemon: Port `9090`, Path `/metrics` (harvesting rate, vector index insertion rate, GCS sync status).

### 7.2. Verifying Scrapes Locally via Port-Forward

```bash
# Port-forward serving endpoint
kubectl port-forward svc/agi-synapse 8000:8000 -n production

# Verify live Prometheus metrics output
curl -s http://127.0.0.1:8000/metrics | grep agi_
```
