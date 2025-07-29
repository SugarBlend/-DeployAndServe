## 📘 Brief Information on Use
Here is a quick rundown of how to run models that build tensorrt inside a Triton inference server container.  
The general steps to run any model based on the core functionality are:
### 1. Custom Handler Implementation

- Create a custom handler class similar to the examples provided in the [`custom`](./overrides) folder.  
- Ensure that your handler adheres to Triton's inference protocol and properly handles input/output processing for your model.

---

### 2. Convert Model Weights

- Use the [deployer container](../deploy2serve/deployment/docker/docker-compose.yaml) to convert PyTorch weights (`*.pt`) into TensorRT format (`*.plan`).  
- Before conversion, select the appropriate configuration file that matches your model architecture.

To launch deploy in same triton container, you must run:
```powershell
./deploy2serve/deployment/docker/deploy.ps1
```

### 3. Set Up Model Repository
- Build the correct folder hierarchy in the Triton model repository:  

  ```
    models/  
    ├── your_model_name/   
    │   ├── 1/   
    │   │   └── model.plan  
    │   └── config.pbtxt  
  ```
  At the moment, only implementations using tensorrt format networks have been tested, as well as using python as a backend for the ensemble.

- Configure the `config.pbtxt` file. You can use the [YOLO example](./models/yolo_trt/config.pbtxt) as a reference.  
  Make sure to define:
  - Input and output tensor names and shapes
  - Data types and formats
  - Dynamic batching parameters (if applicable)
  - GPU execution provider and optimization settings

---

### 4. Launch Containers

- After the export step, copy the generated `.plan` file to the target directory, such as  
  [`triton/models`](./models).
- Start the inference server by running:

```powershell
./deploy2serve/triton/serve.ps1
```

> A more detailed example for Yolo Ultralytics is available [here](./overrides/yolo/README.md)
