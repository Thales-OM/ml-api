import asyncio
import uvicorn
from app_grpc import serve as serve_grpc
from app_fastapi import create_app
from config import MODEL_MANAGER_MAIN_DIR_PATH, TEMPLATES_DIR_PATH, HOST_REST_API, PORT_REST_API, HOST_GRPC, PORT_GRPC
import logging

logging.basicConfig(level=logging.INFO)


async def main():
    # Create the FastAPI app
    app_fastapi = create_app(root_directory=MODEL_MANAGER_MAIN_DIR_PATH, templates_dir_path=TEMPLATES_DIR_PATH)
    config = uvicorn.Config(app_fastapi, host=HOST_REST_API, port=int(PORT_REST_API))
    server = uvicorn.Server(config)

    # Start the gRPC server in the background
    grpc_task = asyncio.create_task(serve_grpc(host=HOST_GRPC, port=PORT_GRPC, root_directory=MODEL_MANAGER_MAIN_DIR_PATH, templates_dir_path=TEMPLATES_DIR_PATH)) 

    await grpc_task

    # Run the server
    await server.serve()
    

if __name__ == "__main__":
    # Get the current event loop and run the main coroutine
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
    asyncio.run(main())
