import os
import tensorrt
import torch
import numpy as np
import onnx
from easytrt import export_onnx, export_engine, TRTBackend, mkdir


model = TRTBackend("/mnt/workdir/testspace/out/resnet18.engine")

r = model(torch.randn(16, 3, 224, 224).cuda())
print(r, r.shape)