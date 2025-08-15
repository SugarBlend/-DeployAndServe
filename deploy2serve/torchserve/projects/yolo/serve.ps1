torch-model-archiver --model-name yolo --version 1.0 --serialized-file .\weights\torchscript\model.pt `
 --handler .\deploy2serve\torchserve\projects\yolo\handler.py `
 --export-path .\deploy2serve\torchserve\projects\yolo\models --force

$env:LOG_LOCATION = "$PWD\deploy2serve\torchserve\projects\yolo\logs"
$env:METRICS_LOCATION = "$PWD\deploy2serve\torchserve\projects\yolo\logs"
torchserve --start --ts-config .\deploy2serve\torchserve\projects\yolo\config.properties --disable-token-auth
