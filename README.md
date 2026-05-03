# Distributed MCP RAG Infrastructure — Technical Report

## 1. Overview

This project implements a fully self-hosted, distributed **MCP (Model Context Protocol) RAG server** backed by a **Qdrant vector database**. The system is deployed on a three-node **Proxmox VE** hypervisor cluster, orchestrated via **k3s Kubernetes**, containerized and distributed through **GitHub Container Registry (GHCR)**, and exposed to the internet via a **Cloudflare Quick Tunnel**. All embedding, indexing, and inference runs on **CPU only**. No GPU is required or used anywhere in the stack.

The system is fully operational and has been validated end-to-end: documents have been ingested into Qdrant, and MCP clients (including Claude.ai) can perform semantic search over the ingested corpus via the public tunnel URL.

---

## 2. Physical Infrastructure

### Proxmox VE Cluster

Three physical machines form a Proxmox VE 8 hypervisor cluster:

| Node Name | Hardware |
|---|---|
| `xochiquetzal` | iMac |
| `quetzalcoatl` | Dell Optiplex |
| `pikachu` | ASUS laptop |

- Clustered via `corosync` / `pvecm`
- Shared cluster state with quorum (3 votes, quorum = 2)
- Automatic VM failover via Proxmox HA

### Storage

- **ZFS** on each node
- **RAID 0** per node
- **local-ZFS** storage pool used for all VM disks

### Known Hardware Issue — e1000e NIC

The `quetzalcoatl` node uses an Intel I219 onboard NIC driven by the `e1000e` kernel driver. Under sustained high-bandwidth workloads, this NIC exhibits a **hardware TX queue stall** (`Detected Hardware Unit Hang`) that takes the entire node offline. The mitigation applied is:

```bash
ethtool -K ens18 tso off gso off gro off tx off
```

This disables hardware TX offload features that trigger the bug. The fix is applied persistently via `/etc/network/interfaces`. Reducing the container image size from **5.15 GB compressed to ~1.18 GB compressed** (by switching to CPU-only PyTorch) significantly reduced the frequency of this event.

---

## 3. Virtual Machines

Four Ubuntu Server 24.04 LTS VMs are deployed across the cluster:

| VM | Role | Network |
|---|---|---|
| `k8s-node-1` | k3s control-plane + worker | VLAN 20 |
| `k8s-node-2` | k3s control-plane + worker | VLAN 20 |
| `k8s-node-3` | k3s control-plane + worker | VLAN 20 |
| `edge-gw` | Internet gateway, NAT, Cloudflare tunnel host | Dual-homed: LAN + VLAN 20 |

Each k8s VM is provisioned with approximately 4 vCPU, 4 GiB RAM, and 70 GB disk. The `edge-gw` VM is lighter and serves only as a network gateway and tunnel host.

> **Note:** The `edge-gw-auth` component (a custom OAuth/API key gateway originally planned to sit in the request path) was **fully eliminated** from the active architecture. It exists as a deployed pod but handles no traffic. All external access goes directly through the Cloudflare tunnel to the MCP server NodePort.

---

## 4. Networking

### Physical Layer

- All nodes connected via **Ethernet** to a **managed VLAN-capable switch**
- No WiFi dependency anywhere in the cluster

### VLAN Design

| VLAN | Purpose |
|---|---|---|
| VLAN 10 | Proxmox cluster communication |
| VLAN 20 | Kubernetes workload network |

### Proxmox Bridges

- `vmbr0` → LAN
- `vmbr0.10` → VLAN 10 (cluster heartbeat)
- `vmbr0.20` → VLAN 20 (workloads)
- `vmbr1` → VLAN 20 workload bridge

### Edge Gateway

The `edge-gw` VM is dual-homed:

- `ens18` → LAN (internet-facing)
- `ens19` → VLAN 20 (workload-facing)

It provides:
- Routing between LAN and VLAN 20
- NAT for outbound internet access from VLAN 20
- Firewall rules isolating the workload network

### Network Isolation

VLAN 20 is isolated from VLAN 10 and the LAN. The only allowed traffic is outbound internet via `edge-gw` and intra-VLAN 20 pod-to-pod communication.

### Request Flow

```
MCP Client (internet)
  → Cloudflare Edge
    → Cloudflare Quick Tunnel (cloudflared on edge-gw)
      → edge-gw
        → k8s NodePort 30900
          → mcp-server pod
            → Qdrant pod (internal service DNS)
```

---

## 5. High Availability

### Proxmox Layer

- 3-node cluster with quorum maintained
- Proxmox HA agent configured for automatic VM migration on node failure
- ZFS replication available between nodes

### Kubernetes Layer (k3s)

- **3-node control-plane** — all three VMs run both control-plane and worker roles
- Pods are distributed across nodes by the scheduler

### MCP Server

- Deployed as a **Deployment with 2 replicas**
- Pods distributed across different nodes where possible
- Liveness and startup probes configured against `/api/healthz`

### Qdrant

- Deployed as a **StatefulSet with 1 replica**
- Persistent volume claim backed by local-ZFS
- Data survives pod restarts; single-node (no Qdrant clustering)

---

## 6. Kubernetes Workloads

### Namespace: `mcp`

| Resource | Kind | Replicas | Purpose |
|---|---|---|---|
| `mcp-server` | Deployment | 2 | MCP + RAG application |
| `qdrant` | StatefulSet | 1 | Vector database |
| `edge-gw-auth` | Deployment | 1 | Present but not in request path |
| `mcp-server` | Service | — | Internal ClusterIP |
| `mcp-server-nodeport` | Service | — | NodePort 30900 |
| `qdrant` | Service | — | Internal ClusterIP (port 6333) |
| `mcp-server-config` | ConfigMap | — | Environment configuration |

### Resource Limits (mcp-server)

```yaml
resources:
  requests:
    cpu: "200m"
    memory: "1Gi"
  limits:
    cpu: "2"
    memory: "4Gi"
```

The memory limit was raised from 2 GiB to 4 GiB to accommodate the embedding model (~500 MB) plus large PDF processing in memory.

---

## 7. MCP RAG Server Application

### Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| MCP framework | FastMCP (streamable HTTP transport) |
| Web server | Uvicorn |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector database client | `qdrant-client` |
| PDF parsing | PyMuPDF (`pymupdf`) |
| Settings management | Pydantic Settings (`pydantic-settings`) |
| Logging | Loguru |
| Package management | `uv` |

### Endpoints

| Path | Method | Purpose |
|---|---|---|
| `/mcp` | POST | MCP Streamable HTTP transport |
| `/api/healthz` | GET | Health check (Qdrant + embedding model) |
| `/api/ingest` | POST | Multipart file upload for document ingestion |

### MCP Tools Exposed

- **`search_documents(query, top_k, source_filter)`** — semantic similarity search over the Qdrant collection, returns ranked text chunks with source paths and scores
- **`list_sources()`** — returns collection statistics and vector store health

---

## 8. Container Image

### Registry

Images are built via **GitHub Actions** on every push and published to **GitHub Container Registry (GHCR)**:

```
ghcr.io/steven-herrera/mcp-rag-server:<tag>
```

### Image Size Reduction

The initial image was **5.15 GB compressed** due to PyTorch resolving to its CUDA variant (including `nvidia-cublas`, `nvidia-cudnn`, `triton`, and related packages) when installed as a transitive dependency of `sentence-transformers` on Linux.

The fix was to explicitly declare `torch` as a direct dependency in `pyproject.toml` and pin it to the CPU-only PyTorch index:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

After this change, `uv lock` resolves `torch==2.x.x+cpu` instead of the CUDA build, dropping the image to **~1.18 GB compressed**. All embedding, indexing, and inference run on CPU only — no GPU is present or required.

### Dockerfile Optimizations

- `tini` used as PID 1 for correct signal handling and zombie reaping
- `snapshot_download` used instead of `SentenceTransformer(...)` at build time — downloads model weights only, skipping the PyTorch warmup pass (reduced model download step from ~22 minutes to ~10 seconds)
- `uv sync` runs as root before `USER appuser` to avoid permission errors writing to the venv
- `.dockerignore` excludes `.git`, `.venv`, `__pycache__`, `tests`, `dist`, and other non-runtime artifacts

---

## 9. Document Ingestion Pipeline

### Supported File Types

| Extension | Parser | Notes |
|---|---|---|
| `.pdf` | PyMuPDF | Text-based PDFs only; no OCR |
| `.txt` | Plain text | UTF-8 |
| `.md`, `.rst` | Plain text | — |
| `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.yaml`, `.json`, `.toml`, `.sh` | Code | Preserves language metadata |

### Large PDF Streaming

Standard PDF ingestion (loading the full document as a single string) caused **OOM kills** when processing the 685-page textbook. The solution is a streaming batch processor in `ingest.py` that processes PDFs in configurable page batches:

- Default `batch_size=20` pages per batch
- Each batch: extract text → chunk → embed → upsert to Qdrant
- `pymupdf.TOOLS.store_shrink(100)` called after each batch to flush MuPDF's internal cache
- Page reference set to `None` after text extraction to release page-level memory

### Qdrant Write Timeout

Large batches of dense chunks caused `httpx.WriteTimeout` on the Qdrant upsert. The client timeout was raised to 120 seconds:

```python
self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=120)
```

### Ingested Corpus

Documents successfully ingested into the `documents` Qdrant collection include the textbook for the distributed systems
course, the lecture slide decks from `DistSys-00-Intro.pdf"` to `DistSys-05.2-SmartNICs.pdf"`, and some lecture
transcripts from the course.

**Total ingested: 150,305 vector chunks**

---

## 10. Cloudflare Quick Tunnel

### Setup

`cloudflared` runs on the `edge-gw` VM as a foreground process:

```bash
cloudflared tunnel --url http://192.168.20.11:30900
```

This creates an outbound-only tunnel from `edge-gw` to Cloudflare's edge network, producing a public HTTPS URL of the form:

```
https://<random-words>.trycloudflare.com
```

No open inbound ports, no public IP, and no firewall changes are required on the cluster.

### What Works

- Full MCP Streamable HTTP protocol (`POST /mcp`)
- Health check endpoint (`GET /api/healthz`)
- Document ingest endpoint (`POST /api/ingest`)
- Claude.ai and ChatGPT remote MCP integration

### Limitations

| Limitation | Detail |
|---|---|
| Ephemeral URL | Subdomain changes every time `cloudflared` restarts |
| No SLA | Explicitly documented as development/testing only |
| No Cloudflare Access | OAuth/Zero Trust policies cannot be attached to Quick Tunnel URLs |
| Concurrent request cap | Hard limit of 200 in-flight requests |
| Upload size cap | Cloudflare free plan enforces 100 MB maximum request body |
| No custom domain | Requires a registered domain for named tunnels |
| Not production-grade | Cloudflare reserves the right to rate-limit or terminate |

---

## 11. Security Posture

### Current State

- MCP server is publicly reachable at the Quick Tunnel URL with no authentication enforced
- The URL is obscure and changes on restart
- `edge-gw-auth` (OAuth + API key gateway) is **not in the request path**

---

## 12. Toolchain & Development Workflow

| Tool | Purpose |
|---|---|
| `uv` | Python package management and virtual environments |
| `ruff` | Linting and formatting |
| `pylint` | Static analysis |
| `pytest` | Unit and integration testing |
| `loguru` | Structured logging |
| `pydantic` / `pydantic-settings` | Data validation and environment config |
| `hadolint` | Dockerfile linting |
| `GitHub Actions` | CI: lint, test, build, push to GHCR |
| `Makefile` | Standardized local dev commands |

---

## 13. Known Limitations

| Area | Limitation |
|---|---|
| PDF ingestion | Text-based only; scanned PDFs require OCR (not implemented) |
| Qdrant | Single-node deployment; no replication or horizontal scaling |
| Embedding | CPU-only; large corpus ingestion is slow (150k chunks took significant time) |
| Quick Tunnel | Ephemeral URL, no auth, 200-request cap |
| Chunk deduplication | UUIDs are currently random (`uuid4`); re-ingesting creates duplicates |
| NIC stability | `e1000e` hardware bug on `quetzalcoatl` can crash the node under heavy network load |
| Qdrant version mismatch | Client 1.17.1 vs server 1.13.2 — minor version skew produces warnings |

---

## 14. Current Status

### Fully Operational

- Proxmox 3-node cluster with HA and ZFS storage
- VLAN-segmented network with edge gateway and NAT
- k3s 3-node HA Kubernetes cluster
- Containerized MCP RAG server (2 replicas, CPU-only)
- Qdrant vector database with 150,305 ingested chunks
- Cloudflare Quick Tunnel with public HTTPS access
- End-to-end MCP protocol (validated with Claude.ai, ChatGPT, and MCP Inspector)
- GitHub Actions CI pipeline pushing to GHCR