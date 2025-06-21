## Brief information on use

### 1. Custom Handler Implementation
- Create a handler class similar to the examples provided in the [custom folder](./custom).  
- Ensure your handler adheres to Triton's inference protocol and handles input/output processing for your model.

### 2. Convert model weights
- Use the [deployer container](../deployment/docker.docker-compose.yaml) to convert PyTorch weights (`*.pt`) to TensorRT format (`*.plan`).  
- Select the appropriate configuration file for your model architecture before conversion.
```powershell
../deployment/docker/deploy.ps1
```

### 3. Set Up Model Repository
- Build the correct folder hierarchy in the Triton model repository:  
models/  
├── your_model_name/   
├── 1/   
├─── model.plan  
└── config.pbtxt  
- Configure the `config.pbtxt` file (see [example](models/yolo_trt/config.pbtxt)) with:
  - Input/output tensor shapes
  - Dynamic batching settings (if needed)
  - GPU acceleration preferences

### 4. Launch containers
It is necessary to copy the weights generated after export and put them in the   
[triton/models](./models) folder, then run the [serve.ps1](./serve.ps1) file.