# ml-api
Web app to train/store/tune ml models

Launch app:
- Only core module: `python app.py`
- With all modules: `docker-compose up --build`

Main addresses:
- Main page: `http://localhost:8000`
    - Swagger: `http://localhost:8000/docs`
    - Grafana: `http://localhost:3000/d/fe39almqz2u4gr`
    - Minio UI: `http://localhost:9001`
    - ClearML UI: `http://localhost:8080`
- gRPC server: `http://localhost:50051`

Example usages (FastApi, gRPC): `/examples`

Author: `ispugin@edu.hse.ru` (Пугин Илья Сергеевич)