# Experiments with Model Serving Technologies

This repository contains examples of working with models across various computer vision (CV) domains.  
It also provides reusable patterns for assembling models in different formats, which serve as the foundation for containerized deployment workflows.

---

## ✅ Prerequisites

The setup has been tested with the following configuration:

- **CUDA:** 12.1.0  
- **cuDNN:** 9.0.0  
- **TensorRT:** 10.10.0.31 (installed via `pyproject.toml`)

---

## 🚀 Deployment

The main entry point for deployment is [`deploy.py`](./deploy2serve/deployment/deploy.py), which should be configured to suit your specific requirements.  
To deploy in a specific environment (e.g., using the [Triton Inference Server](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)),  
the serving plan must be prepared inside the container. This is already accounted for in the project setup.

To launch:

1. Run the PowerShell script [`deploy.ps1`](./deploy2serve/deployment/docker/deploy.ps1) from the command line.
2. Once inside the container (interactive mode), run the deployment script to start the service as expected.

For successful visualization, follow the provided [recommendation](./resources/.gitkeep).  
Files mounted inside the container are linked to the project’s [weights folder](./weights), which is used by default to store models.  
A configuration file [template](./deploy2serve/deployment/deploy_template.json) is available, along with a complete [example](./deploy2serve/deployment/overrides/yolo).

This repository only includes functionality for converting PyTorch models to the following formats:
- ✅ ONNX  
- ✅ TorchScript  
- ✅ TensorRT  
- ✅ OpenVINO

Model loading logic is expected to be provided by the user.

---

## 📦 Supported Model Serving Technologies

The following model serving technologies are currently supported or planned:

- ✅ [TorchServe](./deploy2serve/torchserve/README.md)  
- ✅ [Triton Inference Server](./deploy2serve/triton/README.md)  
- ⏳ **BentoML** *(not yet implemented)*  
- ⏳ **KServe** *(not yet implemented)*  
- ⏳ **Ray Serve** *(not yet implemented)*
