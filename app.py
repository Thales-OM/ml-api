import asyncio
import uvicorn
from prometheus_client import start_http_server, Gauge
from prometheus.prometheus_write import collect_metrics
from app_grpc import serve as serve_grpc
from app_fastapi import create_app
from config import MODEL_MANAGER_MAIN_DIR_PATH, TEMPLATES_DIR_PATH, HOST_REST_API, PORT_REST_API, HOST_GRPC, PORT_GRPC, PORT_PROMETHEUS_SERVER
import logging

logging.basicConfig(level=logging.INFO)


async def main():
    # Start the Prometheus HTTP server
    start_http_server(PORT_PROMETHEUS_SERVER)
    
    # Create the FastAPI app
    app_fastapi = create_app(root_directory=MODEL_MANAGER_MAIN_DIR_PATH, templates_dir_path=TEMPLATES_DIR_PATH)
    config = uvicorn.Config(app_fastapi, host=HOST_REST_API, port=int(PORT_REST_API))
    server = uvicorn.Server(config)
    fastapi_task = asyncio.create_task(server.serve())

    # TODO: Prometheus collects native Python metric but not the app's -> find out why + fix
    # Start the metrics collection in the background
    metrics_collection_task = asyncio.create_task(collect_metrics())  

    # Start the gRPC server in the background
    grpc_task = asyncio.create_task(serve_grpc(host=HOST_GRPC, port=PORT_GRPC, root_directory=MODEL_MANAGER_MAIN_DIR_PATH, templates_dir_path=TEMPLATES_DIR_PATH)) 

    await fastapi_task
    await grpc_task
    await metrics_collection_task

if __name__ == "__main__":
    asyncio.run(main())
