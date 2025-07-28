docker-compose -f deployment/docker/docker-compose.yaml up -d --build
docker-compose -f deployment/docker/docker-compose.yaml exec -it export bash
docker-compose -f deployment/docker/docker-compose.yaml down
