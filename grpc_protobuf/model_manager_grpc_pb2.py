# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: model_manager_grpc.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'model_manager_grpc.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18model_manager_grpc.proto\x12\x12model_manager_grpc\x1a\x1bgoogle/protobuf/empty.proto\"\x1b\n\tFloatList\x12\x0e\n\x06values\x18\x01 \x03(\x02\"\xda\x02\n\nFitRequest\x12.\n\x07X_train\x18\x01 \x03(\x0b\x32\x1d.model_manager_grpc.FloatList\x12\x0f\n\x07y_train\x18\x02 \x03(\x02\x12:\n\x06params\x18\x03 \x03(\x0b\x32*.model_manager_grpc.FitRequest.ParamsEntry\x12\x0c\n\x04loss\x18\x04 \x01(\t\x12\r\n\x05optim\x18\x05 \x01(\t\x12\x41\n\noptim_args\x18\x06 \x03(\x0b\x32-.model_manager_grpc.FitRequest.OptimArgsEntry\x12\x0e\n\x06\x65pochs\x18\x07 \x01(\x05\x1a-\n\x0bParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x30\n\x0eOptimArgsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"?\n\x0ePredictRequest\x12-\n\x06X_test\x18\x01 \x03(\x0b\x32\x1d.model_manager_grpc.FloatList\"\xdc\x01\n\x1a\x45xperimentMetadataResponse\x12\x15\n\rexperiment_id\x18\x01 \x01(\t\x12\x14\n\x0c\x63reated_dttm\x18\x02 \x01(\t\x12\x19\n\x11last_changed_dttm\x18\x03 \x01(\t\x12\x16\n\x0emodel_filename\x18\x04 \x01(\t\x12\x0c\n\x04name\x18\x05 \x01(\t\x12\x1c\n\x14origin_experiment_id\x18\x06 \x01(\t\x12\x1c\n\x14parent_experiment_id\x18\x07 \x01(\t\x12\x14\n\x0ctemplate_flg\x18\x08 \x01(\x08\"4\n\x14\x42\x61sicSuccessResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0b\n\x03msg\x18\x02 \x01(\t\"0\n\x17\x42\x61sicExperimentResponse\x12\x15\n\rexperiment_id\x18\x01 \x01(\t\"&\n\x0fPredictResponse\x12\x13\n\x0bpredictions\x18\x01 \x03(\x02\"^\n\x17ListExperimentsResponse\x12\x43\n\x0b\x65xperiments\x18\x01 \x03(\x0b\x32..model_manager_grpc.ExperimentMetadataResponse\">\n\x17\x42ranchExperimentRequest\x12\x15\n\rexperiment_id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\"0\n\x17SelectExperimentRequest\x12\x15\n\rexperiment_id\x18\x01 \x01(\t\"\x1c\n\x0bSeedRequest\x12\r\n\x05value\x18\x01 \x01(\x05\x32\xdb\x05\n\x10ModelManagerGRPC\x12V\n\x0fListExperiments\x12\x16.google.protobuf.Empty\x1a+.model_manager_grpc.ListExperimentsResponse\x12l\n\x10\x42ranchExperiment\x12+.model_manager_grpc.BranchExperimentRequest\x1a+.model_manager_grpc.BasicExperimentResponse\x12i\n\x10SelectExperiment\x12+.model_manager_grpc.SelectExperimentRequest\x1a(.model_manager_grpc.BasicSuccessResponse\x12R\n\x0eSaveExperiment\x12\x16.google.protobuf.Empty\x1a(.model_manager_grpc.BasicSuccessResponse\x12Q\n\x04Seed\x12\x1f.model_manager_grpc.SeedRequest\x1a(.model_manager_grpc.BasicSuccessResponse\x12O\n\x03\x46it\x12\x1e.model_manager_grpc.FitRequest\x1a(.model_manager_grpc.BasicSuccessResponse\x12R\n\x07Predict\x12\".model_manager_grpc.PredictRequest\x1a#.model_manager_grpc.PredictResponse\x12J\n\x06Health\x12\x16.google.protobuf.Empty\x1a(.model_manager_grpc.BasicSuccessResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'model_manager_grpc_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_FITREQUEST_PARAMSENTRY']._loaded_options = None
  _globals['_FITREQUEST_PARAMSENTRY']._serialized_options = b'8\001'
  _globals['_FITREQUEST_OPTIMARGSENTRY']._loaded_options = None
  _globals['_FITREQUEST_OPTIMARGSENTRY']._serialized_options = b'8\001'
  _globals['_FLOATLIST']._serialized_start=77
  _globals['_FLOATLIST']._serialized_end=104
  _globals['_FITREQUEST']._serialized_start=107
  _globals['_FITREQUEST']._serialized_end=453
  _globals['_FITREQUEST_PARAMSENTRY']._serialized_start=358
  _globals['_FITREQUEST_PARAMSENTRY']._serialized_end=403
  _globals['_FITREQUEST_OPTIMARGSENTRY']._serialized_start=405
  _globals['_FITREQUEST_OPTIMARGSENTRY']._serialized_end=453
  _globals['_PREDICTREQUEST']._serialized_start=455
  _globals['_PREDICTREQUEST']._serialized_end=518
  _globals['_EXPERIMENTMETADATARESPONSE']._serialized_start=521
  _globals['_EXPERIMENTMETADATARESPONSE']._serialized_end=741
  _globals['_BASICSUCCESSRESPONSE']._serialized_start=743
  _globals['_BASICSUCCESSRESPONSE']._serialized_end=795
  _globals['_BASICEXPERIMENTRESPONSE']._serialized_start=797
  _globals['_BASICEXPERIMENTRESPONSE']._serialized_end=845
  _globals['_PREDICTRESPONSE']._serialized_start=847
  _globals['_PREDICTRESPONSE']._serialized_end=885
  _globals['_LISTEXPERIMENTSRESPONSE']._serialized_start=887
  _globals['_LISTEXPERIMENTSRESPONSE']._serialized_end=981
  _globals['_BRANCHEXPERIMENTREQUEST']._serialized_start=983
  _globals['_BRANCHEXPERIMENTREQUEST']._serialized_end=1045
  _globals['_SELECTEXPERIMENTREQUEST']._serialized_start=1047
  _globals['_SELECTEXPERIMENTREQUEST']._serialized_end=1095
  _globals['_SEEDREQUEST']._serialized_start=1097
  _globals['_SEEDREQUEST']._serialized_end=1125
  _globals['_MODELMANAGERGRPC']._serialized_start=1128
  _globals['_MODELMANAGERGRPC']._serialized_end=1859
# @@protoc_insertion_point(module_scope)
