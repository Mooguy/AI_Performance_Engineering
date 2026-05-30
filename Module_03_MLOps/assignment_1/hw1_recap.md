# HW1 Recap — Distributed GPU Training on Nebius Cloud with SkyPilot

## Task 1 — Docker Image

**Dockerfile** — key addition was `openssh-server` (required by SkyPilot):

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y openssh-server && \
    mkdir -p /var/run/sshd && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    transformers datasets accelerate peft trl bitsandbytes wandb scipy

CMD ["bash"]
```

**Build and push:**

```bash
docker build -t cr.eu-north1.nebius.cloud/e00fe961sn3hefk5fs/nebius-trainer:v4 .
docker push cr.eu-north1.nebius.cloud/e00fe961sn3hefk5fs/nebius-trainer:v4
```

---

## Task 2 — mk8s Cluster

**Create cluster** with public endpoint enabled (via console).

**Get credentials:**

```bash
nebius mk8s cluster get-credentials \
  --id mk8scluster-e00bjba1j23rtgt8t2 \
  --external \
  --kubeconfig ~/.kube/config
```

**Create node group** with CUDA drivers and service account:

```bash
nebius mk8s node-group create \
  --name gpu-nodes \
  --parent-id mk8scluster-e00bjba1j23rtgt8t2 \
  --fixed-node-count 2 \
  --template-resources-platform "gpu-l40s-a" \
  --template-resources-preset "1gpu-16vcpu-64gb" \
  --template-os "ubuntu24.04" \
  --template-gpu-settings-drivers-preset "cuda13.0" \
  --template-boot-disk-type network_ssd \
  --template-boot-disk-size-bytes 107374182400 \
  --template-network-interfaces "[{\"public_ip_address\": {}, \"subnet_id\": \"vpcsubnet-e00pxkszfg9c7pqk7a\"}]" \
  --template-service-account-id serviceaccount-e00ss2sg28jecaae3t
```

**Registry access** — create a group, add the SA, grant viewer role:

```bash
nebius iam group-membership create \
  --parent-id group-e00hb7dzqgx78mw2cz \
  --member-id serviceaccount-e00ss2sg28jecaae3t

nebius iam access-permit create \
  --parent-id group-e00hb7dzqgx78mw2cz \
  --resource-id registry-e00fe961sn3hefk5fs \
  --role viewer
```

**Save node group config (deliverable):**

```bash
nebius mk8s node-group get --id mk8snodegroup-e00qpq3ek1yx33d4j9 --format json | jq '{metadata, spec}' > mk8s-ng-config.json
```

---

## Task 3 — SkyPilot

**Install:**

```bash
uv tool install --with pip "skypilot[nebius]"
```

**Connect to shared API server:**

```bash
sky api login -e "https://public-e00-nedmvk31bcf9dng-ekfca60rz5g6fgf-skypilot.gw.msp.eu-north1.nebius.cloud"
```

**Verify:**

```bash
sky api info
sky check kubernetes
```

---

## Task 4 — train_job.yaml

Key settings that worked:

```yaml
name: nebius-ddp-training

workdir: .

resources:
  infra: k8s/indigo-lizard-cluster-2
  accelerators: "L40S:1"
  memory: "60+"
  image_id: docker:cr.eu-north1.nebius.cloud/e00fe961sn3hefk5fs/nebius-trainer:v4

num_nodes: 2

envs:
  MODEL_ID: "facebook/opt-1.3b"
  TRAIN_SCRIPT: "train.py"
  BLOCK_SIZE: "512"
  PER_DEVICE_TRAIN_BATCH_SIZE: "4"
  PER_DEVICE_EVAL_BATCH_SIZE: "4"
  GRADIENT_ACCUMULATION_STEPS: "1"
  DATALOADER_NUM_WORKERS: "8"
  TOKENIZERS_PARALLELISM: "false"
  NCCL_DEBUG: INFO
  NCCL_DEBUG_SUBSYS: INIT,NET

setup: |
  echo "Setup complete"
  nvidia-smi || true

run: |
  set -euxo pipefail

  MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n 1)
  MASTER_PORT=29500

  torchrun \
    --nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE} \
    --nnodes=${SKYPILOT_NUM_NODES} \
    --node_rank=${SKYPILOT_NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    "${TRAIN_SCRIPT}"
```

---

## Task 5 — Run Training

```bash
sky launch -c ddp-run train_job.yaml
sky logs ddp-run --tail 0 > training_log.txt
sky down ddp-run
```

---

## Key Lessons Learned

- Always enable **public endpoint** when creating the mk8s cluster
- Node group needs **`--template-gpu-settings-drivers-preset cuda13.0`** with `ubuntu24.04` for L40S GPUs
- Docker image needs **`openssh-server`** pre-installed for SkyPilot to work
- Service account needs to be attached to the node group for registry access
- Image path in registry is **without** the `registry-` prefix: `cr.eu-north1.nebius.cloud/e00fe961sn3hefk5fs/...`
