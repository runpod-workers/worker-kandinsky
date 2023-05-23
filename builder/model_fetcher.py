import os
import urllib.request
from kandinsky2 import get_kandinsky2

cache_dir = os.path.expanduser("~/.cache/kandinsky2")
clip_cache_dir = os.path.expanduser("~/.cache/clip")

# Make directory if it doesn't exist
os.makedirs(clip_cache_dir, exist_ok=True)

clip_model_url = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
clip_model_path = os.path.join(clip_cache_dir, "ViT-L-14.pt")

# Download the model
urllib.request.urlretrieve(clip_model_url, clip_model_path)

try:
    get_kandinsky2('cuda', task_type='text2img', model_version='2.1',
                   use_flash_attention=False, cache_dir=cache_dir)
except RuntimeError as err:
    print(err)
