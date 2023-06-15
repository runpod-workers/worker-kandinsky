'''
Contains the handler function that will be called by the serverless.
'''


import os
import torch

from diffusers import DiffusionPipeline

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate


from rp_schemas import INPUT_SCHEMA

pipe_prior = DiffusionPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16)
pipe_prior.to("cuda")

t2i_pipe = DiffusionPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
t2i_pipe.to("cuda")
t2i_pipe.enable_xformers_memory_efficient_attention()


def generate_image(job):
    '''
    Generate an image from text using Kandinsky2
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    generator = torch.Generator(device="cuda")
    if validated_input['seed'] != -1:
        generator.manual_seed(validated_input['seed'])

    # Run inference on the model and get the image embeddings
    image_embeds, negative_image_embeds = pipe_prior(
        validated_input['prompt'],
        validated_input['negative_prompt'],
        generator=generator).to_tuple()

    # List to hold the image URLs
    image_urls = []

    # Create image
    output = t2i_pipe(validated_input['prompt'],
                      image_embeds=image_embeds,
                      negative_image_embeds=negative_image_embeds,
                      height=validated_input['h'],
                      width=validated_input['w'],
                      num_inference_steps=validated_input['num_steps'],
                      guidance_scale=validated_input['guidance_scale'],
                      num_images_per_prompt=validated_input['num_images']).images

    # Save the generated images to files
    os.makedirs(f"/{job['id']}", exist_ok=True)

    for index, image in enumerate(output):
        image_path = os.path.join(f"/{job['id']}", f"{index}.png")
        image.save(image_path)

        # Upload the output image to the S3 bucket
        image_url = rp_upload.upload_image(job['id'], image_path)
        image_urls.append(image_url)

    # Cleanup
    rp_cleanup.clean([f"/{job['id']}"])

    # Singular backward compatibility
    if len(image_urls) == 1:
        return {"image_url": image_urls[0]}
    else:
        return {"images": image_urls}


runpod.serverless.start({"handler": generate_image})
