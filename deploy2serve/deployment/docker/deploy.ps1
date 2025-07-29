docker-compose -f deploy2serve/deployment/docker/docker-compose.yaml up -d --build
docker-compose -f deploy2serve/deployment/docker/docker-compose.yaml exec -it export bash
docker-compose -f deploy2serve/deployment/docker/docker-compose.yaml down
