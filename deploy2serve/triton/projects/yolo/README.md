# YOLO Triton Inference Server — Ensemble and Regular Modes

This example demonstrates the implementation of both the regular and ensemble versions of the service using Triton Inference Server.
Before running, you must complete the steps specified in the **basis of this module**.
After successfully forming the model hierarchy, you can run one of the two available modes:

---

## 1. Ensemble Mode

In this mode, unnecessary conversion between CPU and GPU is avoided, improving efficiency.

### Configuration File

The configuration for running the service in ensemble mode looks like this:

```yaml
fastapi:
  host: 127.0.0.1
  port: 5001

triton:
  host: 127.0.0.1
  port: 8001

protocol: grpc

server:
  module: deploy2serve.triton.projects.yolo.server
  cls: EnsembleYoloTriton

```
> 💡 **Note:** This configuration is fundamental to the current functionality of the service and is valid for any other custom implementation.



### Starting the Service

To start the bounding box marking service, run the following command:

```powershell
python deploy2serve/triton/projects/yolo/server.py --service_config deploy2serve/triton/projects/yolo/configs/ensemble.yaml
```

---

## 2. Regular Mode
### Configuration File

The configuration for running the service in ensemble mode looks like this:

```yaml
fastapi:
  host: 127.0.0.1
  port: 5001

triton:
  host: 127.0.0.1
  port: 8001

protocol: grpc

server:
  module: deploy2serve.triton.projects.yolo.server
  cls: RegularYoloTriton

```
### Starting the Service

To start the bounding box marking service, run the following command:

```powershell
python deploy2serve/triton/projects/yolo/server.py --service_config deploy2serve/triton/projects/yolo/configs/regular.yaml
```

---
## Viewing Metrics

You can monitor performance metrics via **Grafana**:

1. Open Grafana in your browser:
   [http://localhost:3000](http://localhost:3000)

2. Import the pre-made [dashboard](../../docker/grafana/grafana-dash.json)

3. Set the `$job` parameter to `triton`.

You should now see performance graphs related to the Triton server.

---

## Interacting with the Service

Visit the service documentation at (default as per config): [http://127.0.0.1:5001/docs](http://127.0.0.1:5001/docs)
You’ll see the **Swagger UI** with available endpoints.

---

### Usage Example

- In the `model_name` field of the `/predict` endpoint, specify the name of the model from the `../../models` directory.
- Upload a **video** or **image** file.
- After processing, a `.pickle` file containing the **frame-by-frame annotation results** will be available.
- You can **download** the results by clicking the corresponding **download** button.
