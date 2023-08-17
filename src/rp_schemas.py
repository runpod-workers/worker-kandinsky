'''
RunPod | Kandinsky | Schemas
'''

INPUT_SCHEMA = {
    
    'prompt': {
        'type': str,
        'required': True,
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': ""
    },
    'negative_prior_prompt': {
        'type': str,
        'required': False,
        'default': ""
    },
    'negative_decoder_prompt': {
        'type': str,
        'required': False,
        'default': ""
    },
    'num_steps': {
        'type': int,
        'required': False,
        'default': 100
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 4
    },
    'h': {
        'type': int,
        'required': False,
        'default': 768
    },
    'w': {
        'type': int,
        'required': False,
        'default': 768
    },
    'sampler': {
        'type': str,
        'required': False,
        'default': 'ddim'
    },
    'prior_cf_scale': {
        'type': float,
        'required': False,
        'default': 4
    },
    'prior_steps': {
        'type': str,
        'required': False,
        'default': "5"
    },
    'seed': {
        'type': int,
        'required': False,
        'default': -1
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1
    },
    'strength': {
        'type': float,
        'required': False,
        'default': 0.2
    },
    'init_image': {
        'type': str,
        'required': False,
        'default': ""
    },

    # Included for backwards compatibility
    'batch_size': {
        'type': int,
        'required': False,
        'default': 1
    }
}
