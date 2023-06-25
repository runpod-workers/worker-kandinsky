# builder/model_fetcher.py

import torch
from diffusers import DiffusionPipeline

def get_kandinsky_pipelines():
    pipe_prior = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", 
                                                   torch_dtype=torch.float16)

    t2i_pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", 
                                                 torch_dtype=torch.float16)

    return pipe_prior, t2i_pipe

if __name__ == "__main__":
    get_kandinsky_pipelines()