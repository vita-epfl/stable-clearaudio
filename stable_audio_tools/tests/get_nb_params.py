import json
from stable_audio_tools.models import create_model_from_config

cfg_path = "stable_audio_tools/configs/model_configs/audio_restoration/rectified_flow_latent.json"
with open(cfg_path) as f:
    cfg = json.load(f)

m = create_model_from_config(cfg)

def count_params(module):
    return sum(p.numel() for p in module.parameters())

total = count_params(m)
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)

# Optional breakdown
core = getattr(m, "model", m)
vae = getattr(m, "pretransform", None)

print(f"Total params: {total:,}")
print(f"Trainable params: {trainable:,}")
print(f"Diffusion core params: {count_params(core):,}")
if vae is not None:
    print(f"Pretransform (VAE) params: {count_params(vae):,}")