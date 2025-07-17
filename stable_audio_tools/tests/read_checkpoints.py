# This script is used to read a checkpoint and print the first 20 keys
import torch, sys, json, os, itertools
ckpt='stable_audio_tools/output/checkpoints/stable-clearaudio-cold-strong-intense-equalizer/magic-dust-2/checkpoints/epoch=613-step=1228-train_loss=0.000000.ckpt'
if not os.path.exists(ckpt):
    print('Checkpoint not found')
    sys.exit(0)
state_dict = torch.load(ckpt, map_location='cpu', weights_only=True)['state_dict']
print('Total keys:', len(state_dict))
first = list(itertools.islice(state_dict.keys(), 20))
print(first)