import asyncio
import grpc
from concurrent import futures
from grpc_protobuf.model_manager_grpc_pb2_grpc import ModelManagerGRPCServicer, add_ModelManagerGRPCServicer_to_server
from grpc_protobuf.model_manager_grpc_pb2 import PredictResponse, BasicExperimentResponse, BasicSuccessResponse, ExperimentMetadataResponse, ListExperimentsResponse
from resources.model_manager import ModelManager
from typing import Optional
import logging
from uuid import UUID
from config import HOST_GRPC, PORT_GRPC, MODEL_MANAGER_MAIN_DIR_PATH, TEMPLATES_DIR_PATH
from datetime import datetime, date


class ModelManagerGRPC(ModelManagerGRPCServicer):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    async def ListExperiments(self, request, context):
        try:
            experiments_on_disk = self.model_manager.get_experiments_on_disk()
            experiments = []
            for experiment_id, experiment_data in experiments_on_disk.items():
                metadata = experiment_data['metadata']
                experiment = ExperimentMetadataResponse(
                    experiment_id=str(experiment_id),
                    created_dttm=self._datetime_to_iso(metadata.get('created_dttm', '')),
                    last_changed_dttm=self._datetime_to_iso(metadata.get('last_changed_dttm', '')),
                    model_filename=metadata.get('model_filename', ''),
                    name=metadata.get('name', ''),
                    origin_experiment_id=str(metadata.get('origin_experiment_id') or ""),
                    parent_experiment_id=str(metadata.get('parent_experiment_id') or ""),
                    template_flg=metadata.get('template_flg', False)
                )
                logging.debug(f"Constructed ExperimentMetadataResponse:\n {experiment}")
                experiments.append(experiment)
            response = ListExperimentsResponse(experiments=experiments)
            logging.debug(f"Final ListExperimentsResponse:\n{response}")
            return response
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return ListExperimentsResponse()  # Optionally return an empty response

    async def BranchExperiment(self, request, context):
        new_experiment_id = self.model_manager.branch_experiment(
            base_experiment_id=UUID(request.experiment_id), 
            name=request.name
        )
        return BasicExperimentResponse(experiment_id=str(new_experiment_id))

    async def SelectExperiment(self, request, context):
        self.model_manager.select_experiment(experiment_id=UUID(request.experiment_id))
        return BasicSuccessResponse(success=True)

    async def SaveExperiment(self, request, context):
        if not self.model_manager.get_current_experiment():
            context.set_details("No experiment currently selected to save")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return BasicSuccessResponse(success=False)
        self.model_manager.save_experiment()
        return BasicSuccessResponse(success=True)

    async def Seed(self, request, context):
        try:
            self.model_manager.seed(value=request.value)
            return BasicSuccessResponse(success=True)
        except:
            return BasicSuccessResponse(success=False)

    async def Fit(self, request, context):
        X_train = [[x_ij for x_ij in x_i.values] for x_i in request.X_train]
        self.model_manager.fit(
            X_train,
            request.y_train,
            request.params,
            request.loss,
            request.optim,
            request.optim_args,
            request.epochs
        )
        return BasicSuccessResponse(success=True)

    async def Predict(self, request, context):
        X_test = [[x_ij for x_ij in x_i.values] for x_i in request.X_test]
        predictions = self.model_manager.predict(X_test)
        return PredictResponse(predictions=predictions)

    async def Health(self, request, context):
        return BasicSuccessResponse(success=True)

    def _datetime_to_iso(self, dt):
        if isinstance(dt, (datetime, date)):
            return dt.isoformat()
        return dt

async def serve(host: str, port: int, root_directory: str, templates_dir_path: Optional[str] = None):
    model_manager = ModelManager(root_directory=root_directory, templates_dir_path=templates_dir_path)
    server = grpc.aio.server()
    add_ModelManagerGRPCServicer_to_server(ModelManagerGRPC(model_manager), server)
    server.add_insecure_port(f'{host}:{port}')
    await server.start()
    logging.info(f"gRPC server is running on port {port}")
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve(host=HOST_GRPC, port=int(PORT_GRPC), root_directory=MODEL_MANAGER_MAIN_DIR_PATH, templates_dir_path=TEMPLATES_DIR_PATH))