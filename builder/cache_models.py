'''
Fetches and caches the Kandinsky models.
'''

import torch
from diffusers import DiffusionPipeline, KandinskyV22PriorPipeline, KandinskyV22Pipeline

def get_kandinsky_pipelines():
    # Kandinsky 2.1 pipelines
    pipe_prior_2_1 = DiffusionPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16)
    t2i_pipe_2_1 = DiffusionPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)

    # Kandinsky 2.2 pipelines
    pipe_prior_2_2 = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16)
    t2i_pipe_2_2 = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)

    return (pipe_prior_2_1, t2i_pipe_2_1), (pipe_prior_2_2, t2i_pipe_2_2)

if __name__ == "__main__":
    get_kandinsky_pipelines()
