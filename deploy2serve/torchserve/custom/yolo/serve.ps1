torch-model-archiver --model-name yolo --version 1.0 --serialized-file deployment/weights/torchscript/model.pt `
 --handler torchserve/custom/yolo/handler.py --export-path torchserve/custom/yolo/models --force
$env:LOG_LOCATION = "$PWD/torchserve/custom/yolo/logs"; $env:METRICS_LOCATION = "$PWD/torchserve/custom/yolo/logs"; `
 torchserve --start --ts-config torchserve/custom/yolo/config.properties --disable-token-auth
