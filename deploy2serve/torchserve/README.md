# 🚀 TorchServe & YOLO Deployment Guide

## 1️⃣ Base TorchServe Commands

```powershell
# Check if the TorchServe server is running
curl http://localhost:8080/ping

# List available models served by TorchServe
curl http://localhost:8081/models

# Stop the TorchServe server
torchserve --stop
```

## 2️⃣ YOLO Model Serving Workflow

### Step 1: Generate TorchScript Weights

Run the deployment script using the configuration file to export the YOLO model to TorchScript format:

```powershell
python .\deploy2serve\deployment\deploy.py --deploy_config .\deploy2serve\deployment\overrides\dynamic.yml
```

## Step 2: Convert TorchScript Model for TorchServe (.mar)

Prepare the `.mar` file (model archive) needed by TorchServe by running the following PowerShell script:

```powershell
.\deploy2serve\torchserve\overrides\yolo\serve.ps1
```

## Step 3: Start Grafana & Prometheus

To monitor your TorchServe instance with **Grafana** and **Prometheus**, start all necessary services using the provided PowerShell script:

```powershell
.\deploy2serve\torchserve\serve.ps1
```
### ✅ After Running

- **TorchServe Management API**: [http://localhost:8081/models](http://localhost:8081/models)
- **Prometheus UI**: [http://localhost:9090](http://localhost:9090)
- **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000)
