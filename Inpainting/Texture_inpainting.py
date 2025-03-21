import torch
import sys
import argparse
import numpy as np
import torch.nn as nn
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
from clip_w_local import clip
import time
def setup_args(parser):
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--min", type=float, required=True,
        help="The path to the lama checkpoint.",
    )

    parser.add_argument(
        "--max", type=float, required=True,
        help="The path to the lama checkpoint.",
    )
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class ZSLCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        clip_model = clip_model.cuda()
        prompt_text = ['a photo of a '+name.replace("_", " ") for name in classnames]
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompt_text]).cuda()
        self.embedding = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)

    def forward(self, image):
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))

        prompts =  self.embedding.type(self.dtype)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        logits_local = logit_scale * local_image_features @ text_features.T

        return logits, logits_local


def load_clip_to_cpu():
    backbone_name = 'ViT-B/16'
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    classnames  = []
    idx2clsname = {}
    # classnames = OrderedDict()
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            classname = line[2]
            idx2clsname[line[0]] = line[1]
            # classname = " ".join(line[1:])
            classnames.append(classname)
    return classnames,idx2clsname
import os


def get_image_paths(root_folder):
    image_paths = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    root = args.output_dir
    os.makedirs(os.path.join(root, 'texture', 'inpaint'), exist_ok=True)
    os.makedirs(os.path.join(root, 'texture', 'visualization'), exist_ok=True)
    os.makedirs(os.path.join(root, 'texture', 'inpaint','top'), exist_ok=True)
    os.makedirs(os.path.join(root, 'texture', 'inpaint','low'), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = load_clip_to_cpu().cuda()
    text_file = "./classnames.txt"
    classnames,id2clsname = read_classnames(text_file)
    CLIP_model = ZSLCLIP(classnames,clip_model)
    for name, param in CLIP_model.named_parameters():
        param.requires_grad_(False)

    folder_path = root+"/mask/"
    image_files = get_image_paths(folder_path)

    print(image_files)
    kernel = np.ones((12,12), np.uint8)  
    for imgname in image_files:

        start_time = time.time()
        imgname = imgname.split('/')[-1]
        mask = load_img_to_array(os.path.join(root+"/mask/",imgname))

        img = load_img_to_array(os.path.join(root+"/raw/", imgname[0:-4]+'.JPEG'))
        mask1 = mask[:,:,0]
        patch_size = 32
        mask1 = torch.tensor(mask1)
        mask_patches = mask1.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        patch_counts = (mask_patches == 255).sum(dim=(2, 3))

        patch_counts = patch_counts == 32*32
        patch_counts = patch_counts.view(-1).numpy()
        mask1 = mask1.numpy()
        true_indices = np.where(patch_counts)[0]

        if true_indices.size > 0:
            random_true_index = np.random.choice(true_indices)
            print("random choose:", random_true_index)
        else:
            print("no region")
            continue

        mask2 = np.full((224, 224), 255)
        region_size=32
        i = random_true_index
        mask2[(i * region_size // 224) * region_size:(i * region_size // 224) * region_size + region_size,
        i * region_size % 224:(i + 1) * region_size % 224] = 0

        if np.min(mask2) == 255:
            pass

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=0)
            img = np.transpose(img, (1, 2, 0))
        try:
            img_inpainted_d = inpaint_img_with_lama(
                img, mask2, args.lama_config, args.lama_ckpt, device=device)

            mask2 = np.stack((mask2,) * 3, axis=-1)
            horizontal_concat = np.concatenate((img, img_inpainted_d, mask2), axis=1)

            img_inpainted_p = os.path.join(root+"/texture/inpaint/",
                                           imgname)
            save_array_to_img(img_inpainted_d, img_inpainted_p)

            visual = os.path.join(root+"/texture/visualization/",
                                           imgname)
            save_array_to_img(horizontal_concat, visual)
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(elapsed_time)
        except:
            pass
