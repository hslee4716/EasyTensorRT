import torch
import numpy as np

from transformers.models.gpt_neox import GPTNeoXTokenizerFast
from time import time

from easytrt import TRTBackend

if __name__ == '__main__':
    tokenizer = GPTNeoXTokenizerFast.from_pretrained('nlpai-lab/kullm-polyglot-12.8b-v2', export=True)
    model = TRTBackend("/home/team_ai/JMJ/nlp_model/kullm_dev/HS/sav_onnx/decoder_model.engine")    

    for i in range(100):
        t = time()
        inputs = tokenizer("why people got cold?", return_tensors="pt")
        input_ids = inputs.input_ids
        masks = torch.ones_like(input_ids)
        input_ids, masks
        output = model([input_ids, masks])[-1]
        output_np = output.detach().cpu().numpy()
        output_ids = np.argmax(output_np, 2)
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(time()-t)