## 📘 Brief Information on Use

### 1. Custom Handler Implementation

- Create a custom handler class similar to the examples provided in the [`custom`](./custom) folder.  
- Ensure that your handler adheres to Triton's inference protocol and properly handles input/output processing for your model.

---

### 2. Convert Model Weights

- Use the [deployer container](../deployment/docker/docker-compose.yaml) to convert PyTorch weights (`*.pt`) into TensorRT format (`*.plan`).  
- Before conversion, select the appropriate configuration file that matches your model architecture.

```powershell
../deployment/docker/deploy.ps1
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

- Configure the `config.pbtxt` file. You can use the [YOLO example](./custom/yolo/models/yolo_trt/config.pbtxt) as a reference.  
  Make sure to define:
  - Input and output tensor names and shapes
  - Data types and formats
  - Dynamic batching parameters (if applicable)
  - GPU execution provider and optimization settings

---

### 4. Launch Containers

- After the export step, copy the generated `.plan` file to the target directory, such as  
  [`triton/models`](./custom/yolo/models).
- Start the inference server by running:

```powershell
./serve.ps1
```