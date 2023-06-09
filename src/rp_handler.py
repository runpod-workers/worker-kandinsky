'''
Contains the handler function that will be called by the serverless.
'''


import os
import torch

from diffusers import DiffusionPipeline, KandinskyImg2ImgPipeline, KandinskyPriorPipeline
from diffusers.utils import load_image

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate


from rp_schemas import INPUT_SCHEMA

pipe_prior = DiffusionPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16).to("cuda")

t2i_pipe = DiffusionPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16).to("cuda")
t2i_pipe.enable_xformers_memory_efficient_attention()

i2i_prior = KandinskyPriorPipeline(**pipe_prior.components).to("cuda")
i2i_pipe = KandinskyImg2ImgPipeline(**t2i_pipe.components).to("cuda")

def _setup_generator(seed):
    generator = torch.Generator(device="cuda")
    if seed != -1:
        generator.manual_seed(seed)
    return generator


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        image_url = rp_upload.upload_image(job_id, image_path)
        image_urls.append(image_url)
    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def generate_image(job):
    '''
    Generate an image from text using Kandinsky2
    '''
    job_input = job["input"]

    # Backwards compatibility
    if job_input.get('batch_size') is not None:
        job_input['num_images'] = job_input['batch_size']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    generator = _setup_generator(validated_input['seed'])

    # Run inference on the model and get the image embeddings
    image_embeds, negative_image_embeds = pipe_prior(
        validated_input['prompt'],
        validated_input['negative_prompt'],
        generator=generator).to_tuple()

    # Check if an input image is provided, determining if this is text2image or image2image
    init_image = None
    if job_input.get('init_image', None) is not None:
        init_image = load_image(job_input['init_image'])
    
    # List to hold the image URLs
    image_urls = []

    if init_image is None:
        # Create text2image
        output = t2i_pipe(validated_input['prompt'],
                        image_embeds=image_embeds,
                        negative_image_embeds=negative_image_embeds,
                        height=validated_input['h'],
                        width=validated_input['w'],
                        num_inference_steps=validated_input['num_steps'],
                        guidance_scale=validated_input['guidance_scale'],
                        num_images_per_prompt=validated_input['num_images']).images
    else:
        # Create image2image
        output = i2i_pipe(
                        validated_input["prompt"],
                        image=init_image,
                        image_embeds=image_embeds,
                        negative_image_embeds=negative_image_embeds,
                        height=validated_input['h'],
                        width=validated_input['w'],
                        num_inference_steps=validated_input['num_steps'],
                        strength=validated_input['strength']).images
        

    image_urls = _save_and_upload_images(output, job['id'])

    return {"image_url": image_urls[0]} if len(image_urls) == 1 else {"images": image_urls}


runpod.serverless.start({"handler": generate_image})
