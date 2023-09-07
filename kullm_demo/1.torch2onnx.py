# load_model, tokenizer
from transformers import AutoModelForCausalLM
from transformers.models.gpt_neox import GPTNeoXTokenizerFast
from time import time

import torch
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="5"
os.environ["CUDA_MODULE_LOADING"]="LAZY"

base_model ="nlpai-lab/kullm-polyglot-12.8b-v2"
model = AutoModelForCausalLM.from_pretrained(
base_model,
torch_dtype=torch.float16,
low_cpu_mem_usage=True,
).cuda()

tokenizer = GPTNeoXTokenizerFast.from_pretrained('nlpai-lab/kullm-polyglot-12.8b-v2', export=True)


#model_test
# model output
token = tokenizer("black top big t-shirts billie eilish", return_tensors="pt")
r = model(token.input_ids.cuda())
output = tokenizer.batch_decode(np.argmax(r[0].detach().cpu().numpy(),2), skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output)
# prompt = "안녕하십니까?"
# token = tokenizer(prompt, return_tensors="pt").cuda()

#save model
# model.save_pretrained("/home/team_ai/JMJ/nlp_model/kullm_dev/HS/sav_bin2")

# after this, run optimum_cli export
# $ optimum-cli export onnx --model <saved_bin_folder_pth> --task text-generation <folder_path_for_sav_ONNX>
# os.system("optimum-cli export onnx --model /home/team_ai/JMJ/nlp_model/kullm_dev/HS/sav_bin2 --task text-generation /home/team_ai/JMJ/nlp_model/kullm_dev/HS/sav_onnx2")
