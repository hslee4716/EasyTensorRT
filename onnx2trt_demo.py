import os
from torchvision.models import resnet18
from easytrt import export_onnx, export_engine, mkdir

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_MODULE_LOADING"]="LAZY"
# os.environ["CUDA_VISIBLE_DEVICES"]="5"

model = resnet18().to('cuda')

out_f = "./out"
mkdir(out_f)

dshape = {'input':{0:'batch'}, 'output':{0:'batch'}}
# export onnx
f = export_onnx(model,
                dynamic=True,
                out_f=f"{out_f}/resnet18.onnx",
                input_shape=[1, 3, 224, 224],
                dshape=dshape,
                device='cuda',)


dshape = {'input':{'min':(1,3,224,224), 'opt':(8,3,224,224), 'max':(16,3,224,224)},
            'output':{'min':(1,1000), 'opt':(8,1000), 'max':(16,1000)}}

export_engine(f, workspace=10, verbose='info', fp16=True, 
              dshapes=dshape, dynamic=True)