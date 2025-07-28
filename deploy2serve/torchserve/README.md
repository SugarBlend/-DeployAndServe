### 1. Packing the model into a .mar archive
```powershell
torch-model-archiver `
    --model-name model `
    --version 1.0 `
    --serialized-file model.pt `
    --handler handler.py `
    --export-path models `
    --extra-files model.py
```

### 2. Launch TorchServe
```powershell
# Common launch
torchserve --start `
    --model-store .\models `
    --models model=model.mar

# With configuration file
torchserve --start `
    --ts-config .\config.properties --disable-token-auth --foreground
```

### 3. Base commands
```powershell
# Check server status
curl http://localhost:8080/ping

# Show list of available models
curl http://localhost:8081/models

# Stop server
torchserve --stop
```
