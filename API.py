import os
import torch
from easytrt import export_onnx, export_engine, TRTBackend

# set GPU env
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_MODULE_LOADING"]="LAZY"
# os.environ["CUDA_VISIBLE_DEVICES"]="5"

# example model
model = torch.nn.Conv2d(3, 64, 3, 1, 1).to('cuda')

# export model to onnx
# input_shape : test input.
# dshape : dynamic shape axis, key's name can be customed
out_f = export_onnx(model, dynamic=True, out_f="./test_model.onnx",
                    input_shape=[1,3,224,224], device='cuda',
                    dshape={'input':{0:'batch'}, 'output':{0:'batch'}})

# export engine from onnx
# when use dynamic shapes, need dshape parameter.
# dshape comes from the box marked above.
dshape = {"input":{"min":(1,3,224,224), "opt":(8,3,224,224), "max":(16,3,224,224)},
          "output":{"min":(1,64,224,224), "opt":(8,64,224,224), "max":(16,64,224,224)}}

export_engine(out_f, workspace=10, verbose='info', 
              fp16=True, dynamic=True, dshapes=dshape,)


# infer TensorRT Engine
engine_path = "./test_model.engine"
model = TRTBackend(engine_path)
result = model(torch.ones((8,3,224,224)))
print(result, result.shape)