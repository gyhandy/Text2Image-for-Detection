# make sure you're logged in with `huggingface-cli login`
import argparse
import json, os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--num_gen_images_per_caption", "-n", type=int, default=20)
    parser.add_argument("--caption_json", default="./data", help="if not '', will only generate DallE images from this json, use `idx` and `scene` to select which to generate")
    parser.add_argument("--num_clusters", default=100, type=int, help="when using RuDalle, split all captions into `num_clusters` chunk and let each machine handle one chunk only")

    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    if 'PT_DATA_DIR' in os.environ:
        args.output_dir = os.path.join(os.environ['PT_DATA_DIR'], args.output_dir)
    return args

def batchify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    args = parse_args()

    with open(args.caption_json) as f:
        data = json.load(f)

    key = next(iter(data))
    # key = "JPEGImages"

    data = data[key]

    all_keys = sorted(list(data.keys()))
    all_chunks = np.array_split(all_keys, args.num_clusters)
    chunks = all_chunks[args.idx]

    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_type=torch.float16)
    pipe = pipe.to("cuda")

    n_already_generated = 0
    n_generate_per_cycle = 4
    while n_already_generated < args.num_gen_images_per_caption:
        for id in chunks:
            for cap in data[id]:
                prompts = [cap] * n_generate_per_cycle
                cap = cap[:50] # too long captions will cause path error
                cap = cap.replace('"', "") # server don't like ", will map to %2522
                os.makedirs(os.path.join(args.output_dir, id, cap), exist_ok=True)
                cur_i = len(list(os.listdir(os.path.join(args.output_dir, id, cap)))) + 1
                for prompt_chunk in batchify(prompts, n=args.bsz):
                    x = pipe(prompt_chunk)
                    images = x.images
                    for img in images:
                        cur_i += 1
                        # img.resize((256, 256)).save(os.path.join(args.output_dir, id, cap, f"{cur_i}.png"))
                        img.save(os.path.join(args.output_dir, id, cap, f"{cur_i}.png"))

        n_already_generated += n_generate_per_cycle