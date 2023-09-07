import torch
import tensorrt as trt
import json
import os
import onnx
import numpy as np
from pathlib import Path

latest_opset = max(int(k[14:]) for k in vars(torch.onnx) if 'symbolic_opset' in k) - 1  # opset

def export_onnx(model,
                dynamic=False,
                out_f = "./out.onnx",
                input_shape=[1, 3,1280,1280],
                dshape={'input':{0:'batch', 2:'height', 3:'width'}, 'output0':{0:'batch', 1:'anchor'}},
                device='cuda',
                onnx_opset = None):

    opset_version = onnx_opset if onnx_opset else latest_opset
    f = out_f
    
    if dynamic:
        dynamic = dshape
    
    im = torch.zeros(input_shape).to(device)
    model.eval()
    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset_version,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic or None)
    model_onnx = onnx.load(f)  # load onnx model

    try:
        import onnxsim
        print("export_onnxsim")
        model_onnx, check = onnxsim.simplify(model_onnx)
    except Exception as e:
        print("onnxsim failed :", e)

    onnx.save(model_onnx, f)
    
    return f
    
def export_engine(onnx_file_path, 
                  workspace=6, 
                  verbose='info', 
                  fp16=True, 
                  metadata={},
                  dynamic=False,
                  dshapes = None,
                  out_pth=None):
    # set verbose
    suffix = Path(onnx_file_path).suffix
    trt_logger = trt.Logger()
    if verbose == 'info':
        trt_logger.min_severity = trt.Logger.Severity.INFO
    elif verbose == True:
        trt_logger.min_severity = trt.Logger.Severity.VERBOSE
    else:
        trt_logger.min_severity = trt.Logger.Severity.ERROR

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)
    
    if dshapes and dynamic:
        profile = builder.create_optimization_profile()
        for name, shape in dshapes.items():
            profile.set_shape(name, min=shape["min"], opt=shape["opt"], max=shape["max"])
    
        config.add_optimization_profile(profile)

    
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    
    parser = trt.OnnxParser(network, trt_logger) 
    parser.parse_from_file(onnx_file_path)
    
    if builder.platform_has_fast_fp16 and fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    metadata['fp16'] = fp16
    metadata['dshape'] = dshapes if dynamic else False
    out_pth = out_pth if out_pth else onnx_file_path.replace(suffix, '.engine')
    with builder.build_serialized_network(network, config) as engine, open(out_pth, 'wb') as f:
        metadata = json.dumps(metadata)
        f.write(len(metadata).to_bytes(4, byteorder='little', signed=True))
        f.write(metadata.encode())
        f.write(engine)
    
    return f.name

class TRTBackend:    
    def __init__(self, engine_path, device='cuda',verbose='info', strict_infer=False):
        self.device=device
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.strict_infer = strict_infer
        if verbose == 'info':
            self.logger.min_severity = trt.Logger.Severity.INFO
        elif verbose == True:
            self.logger.min_severity = trt.Logger.Severity.VERBOSE
        else:
            self.logger.min_severity = trt.Logger.Severity.ERROR
            
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime, "Failed to create trt Runtime"
            config_bytes = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
            self.configs = json.loads(f.read(config_bytes).decode('utf-8'))  # read metadata
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine, "Failed to deserialize CUDA engine"
        
        self.context = self.engine.create_execution_context()
        assert self.context, "Failed to create execution context"

        # Setup I/O bindings
        self.input_shapes=[]
        self.inputs = []
        self.outputs = []
        self.data_ptrs = []
        self.input_dshape = {}
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            
            if self.engine.binding_is_input(i):
                cur_binding_shape = self.engine.get_binding_shape(i)
                self.input_shapes.append(cur_binding_shape)
                if -1 in tuple(cur_binding_shape):  # dynamic
                    profile_shape = self.engine.get_profile_shape(0, i)
                    self.input_dshape[i] = profile_shape
                    self.context.set_binding_shape(i, tuple(profile_shape[2]))
                    
            shape = self.context.get_binding_shape(i)
                  
            empty_array = np.empty(shape, dtype=np.dtype(trt.nptype(dtype)))
            empty_array = torch.tensor(empty_array).to(self.device)
            alloc_ptr = int(empty_array.data_ptr())
            binding = {
                "index": i,
                "name": name,
                "data": empty_array,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": alloc_ptr,
            }
            self.data_ptrs.append(alloc_ptr)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        if len(self.input_dshape) > 0:
            self.dynamic=True
        else:
            self.dynamic=False
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.data_ptrs) > 0
    
    def set_context_input_output_shape(self, inputs:list[torch.Tensor]):
        for i in range(len(self.inputs)):
            input_shape = list(inputs[i].shape)
            if input_shape != self.inputs[i]['shape']:
                if self.dynamic:
                    self.inputs[i]['shape'] = input_shape
                    assert self.context.set_binding_shape(i, input_shape)
                else:
                    if self.strict_infer or self.inputs[i]['shape'][0] - input_shape[0] == 0:
                        assert False, f'input shape : {input_shape} not equal with binding shape : {self.inputs[i]["shape"]}:'
                    else:
                        self.lack_batch = self.inputs[i]['shape'][0] - input_shape[0]
                        inputs[i] = torch.cat([inputs[i]]+[torch.zeros_like(inputs[i][0]).unsqueeze(0)]*self.lack_batch, 0)
            inputs[i] = inputs[i].to(self.device)            
            alloc_ptr = inputs[i].data_ptr()
            self.data_ptrs[self.inputs[i]['index']] = int(alloc_ptr)
            self.inputs[i]['allocation'] = int(alloc_ptr)
        
        if self.dynamic:
            for out in self.outputs:
                nshape = self.context.get_binding_shape(out["index"])
                out["shape"] = nshape
                empty_array = np.empty(nshape, dtype=out['dtype'])
                empty_array = torch.tensor(empty_array).to(self.device)
                alloc_ptr = int(empty_array.data_ptr())
                out["data"] = empty_array
                out['allocation'] = alloc_ptr
                self.data_ptrs[out["index"]] = alloc_ptr
        
    def __call__(self, inputs):
        if len(self.inputs) == 1:
            inputs = [inputs]
        assert len(inputs) == len(self.inputs), f"Input length mismatch: input len::{len(inputs)} != require len::{len(self.inputs)}"
        self.set_context_input_output_shape(inputs)
        assert len(inputs) > 0, "Input length must be greater than 0"
        assert len(self.outputs) > 0, "Output length must be greater than 0"
        
        self.context.execute_v2(bindings=self.data_ptrs)
        result = [out['data'] for out in self.outputs]
        return result if len(self.outputs) > 1 else result[0]