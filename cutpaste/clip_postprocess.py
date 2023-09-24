from collections import defaultdict
from concurrent import futures
from pathlib import Path

import json
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel

voc_texts = [
    f"a photo of {obj}"
    for obj in [
        "person",
        "bird", 'cat', 'cow', 'dog', 'horse', 'sheep',
        'aeroplane', 'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'dining table', 'potted plant', 'sofa', "tv/ monitor"
    ]
]

def batchify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

@torch.no_grad()
def get_CLIP_score(caption: str, images: list):
    logits_per_images = []
    for img in batchify(images, 400):
        inputs = processor(text=[caption] + voc_texts, images=img, return_tensors="pt", padding=True).to("cuda")
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        logits_per_images.append(logits_per_image)
    # probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return torch.cat(logits_per_images, dim=0)

def scores_for_one_caption(caption: Path):
    keep_files = 30
    images = []
    for image in caption.iterdir(): # eg 1.png
        try:
            images.append(Image.open(image))
        except:
            pass # weird generation error
    scores = get_CLIP_score(caption.stem, images) # (#images, 22)

    # 1. select top keep_files*2 lowest consistent_with_voc_labels
    consistent_with_voc_labels = scores[:, 1:].max(1).values
    double_keep_files = min(keep_files * 2, scores.size(0))
    _, indices = torch.topk(-consistent_with_voc_labels.squeeze(), min(double_keep_files, scores.size(0)))
    # 2. select top keep_files highest consistent_with_caption
    consistent_with_caption = scores[indices, 0]
    _, indices = torch.topk(consistent_with_caption, keep_files)
    selected_images = [
        images[i].filename.split("/")[-1]
        for i in indices.detach().cpu().numpy().tolist()
    ]
    return caption.stem, selected_images

def sort_images(images):
    return sorted(images, key=lambda x: int(x.split(".png")[0]))

if __name__ == "__main__":
    pwd = Path(__file__).parent.resolve()
    # root = pwd / "artifact" / "syn" / "voc_1k_bg" / "diffusion_wordnet_v1-10shot"
    # root = pwd.parent / "data" / "voc2012" / "background" / "critical_distractor_v1-10shot"
    # root = pwd.parent / "data" / "voc2012" / "background" / "critical_distractor_v1-10shot"
    # root = pwd.parent / "data" / "voc2012" / "background" / "critical_wordnet_diffusion_v2-10shot"
    # root = pwd.parent / "data" / "voc2012" / "background" / "diffusion_v1_600each"
    # root = pwd.parent / "data" / "voc2012" / "background" / "critical_wordnet_diffusion_v2-1shot"
    # root = pwd.parent / "data" / "voc2012" / "background" / "critical_wordnet_diffusion_v2-10shot_refined"
    root = pwd.parent / "data" / "voc2012" / "background" / "critical_context_only-10shot"
    # root = pwd.parent / "data" / "voc2012" / "background" / "context_augment"

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    to_save = defaultdict(lambda: {})

    # for object in ['empty living room', 'railway without train', 'stable']:
    for object in tqdm(root.iterdir(), total=len(list(root.iterdir()))): # eg a bicycle
        object = root / object
        if not object.name.endswith(".jpg"):
            continue
        captions = list(object.iterdir()) # eg a bicyle in a black background
        with futures.ThreadPoolExecutor(80) as executor:
            res = executor.map(scores_for_one_caption, captions)
            for caption, images in res:
                to_save[object.stem][caption] = sort_images(images)
    # with open(root / "clip_postprocessed.json", "w") as f:
    #     json.dump(to_save, f, indent=4)
    with open("clip_postprocessed.json", "w") as f:
        json.dump(to_save, f, indent=4)