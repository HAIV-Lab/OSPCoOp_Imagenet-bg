import torch
from PIL import Image
import torch.nn as nn
from clip_w_local import clip
import sys
import argparse
import os
import shutil
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
    classnames  = []
    idx2clsname = {}
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            classname = line[2]
            idx2clsname[line[0]] = line[1]
            classnames.append(classname)
    return classnames,idx2clsname
def setup_args(parser):
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    root = args.output_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = load_clip_to_cpu().cuda()
    text_file = "classnames.txt"
    classnames,id2clsname = read_classnames(text_file)
    _, preprocess = clip.load("ViT-B/16", device="cpu")
    CLIP_model = ZSLCLIP(classnames,clip_model)
    for name, param in CLIP_model.named_parameters():
        param.requires_grad_(False)
    folder_path = root+"/inpaint/"

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    for img in image_files:
        image = preprocess(Image.open(os.path.join(folder_path,img))).unsqueeze(0).to(device)
        clsid =img.split('_')[0]
        clsname = int(id2clsname[clsid])-1
        with torch.no_grad():
            output, output_local = CLIP_model(image)
        logit = output.squeeze(0)[clsname]
        print('inp:',logit)
        image = preprocess(Image.open(os.path.join(root+"/raw/", img[0:-4]+'.JPEG'))).unsqueeze(0).to(device)
        clsname = int(id2clsname[clsid]) - 1
        with torch.no_grad():
            output, output_local = CLIP_model(image)
        logit_new = output.squeeze(0)[clsname]
        print('raw:', logit_new)
        new_name = img[0:-4]+ '_' + str(float(logit_new)-float(logit)) + '.JPEG'
        shutil.copyfile(os.path.join(folder_path,img), os.path.join(root+"/score_inpaint/",new_name))
