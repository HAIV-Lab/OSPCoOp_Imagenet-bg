import os
import random
import time
import shutil
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from dassl.data.datasets import Datum

def apply_id_aug(train_root, mask_thre=0.5, shots=1, seed=0, id_aug_options='1,2,3', 
                 id_aug_times='1,1,1', id_aug_rate='0.5,0.5,0.5'):
    '''
    augment ID samples for OOD detection
    
    Args:
        train_root: root directory for training data
        mask_thre: threshold for mask area ratio
        shots & seed: used for sampling
        id_aug_options: ID augmentation options, e.g., '1,2,3'
        id_aug_times: number of times for each option, e.g., '1,1,1'
        id_aug_rate: augmentation rate for each option, e.g., '0.5,0.5,0.5'
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Loading class names
    classnames_path = '/data/hdd/datasets/ImageNet/classnames.txt'
    id2clsname = load_imagenet_classnames(classnames_path)
    
    # set directories
    inpainted_dir = './data/' + train_root + '/inpaint'
    inpaint_thre = 5
    texture_dir = './data/' + train_root + '/texture/inpaint/top'
    image_dir = './data/' + train_root + '/raw'
    mask_dir = './data/' + train_root + '/mask'
    score_inpaint_dir = './data/' + train_root + '/score_inpaint'
    
    # check if directories exist
    required_dirs = [image_dir, mask_dir]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"directories do not exist: {dir_path}")
            return
    
    all_image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png', 'JPEG'))]
    files = []
    if os.path.exists(score_inpaint_dir):
        for filename in os.listdir(score_inpaint_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.JPEG')):
                try:
                    scores = float(os.path.splitext(filename)[0].split('_')[2])
                    if scores > inpaint_thre:
                        files.append(filename.split('_')[0] + '_' + filename.split('_')[1])
                except (IndexError, ValueError) as e:
                    print(f"failed to read {filename}: {e}")
    
    print(f"find {len(files)} images with mask")
    
    # filter image files based on the score
    image_files = [f for f in all_image_files if f.split('_')[0] + '_' + f.split('.')[0].split('_')[1] in files]
    print(f"find {len(image_files)} candidates for ID augmentation")
    
    # set save directory
    save_dir = f"{id_aug_options}_{id_aug_times}_{id_aug_rate}_{mask_thre}_{int(shots)}-shot_seed-{seed}"
    # base_save_path = './ID_Aug_Samples'
    base_save_path = './demo1'
    save_path = os.path.join(base_save_path, save_dir)
    
    os.makedirs(base_save_path, exist_ok=True)
    
    # check if save directory already exists
    if os.path.exists(save_path) and os.listdir(save_path):
        print(f"directory {save_path} already exists and is not empty, skipping augmentation")
        return
    elif os.path.exists(save_path):
        print(f"directory {save_path} already exists, but is empty, removing it")
        os.rmdir(save_path)

    os.makedirs(save_path, exist_ok=True)
    
    id_aug_options = list(map(int, id_aug_options.split(',')))
    id_aug_times = list(map(int, id_aug_times.split(',')))
    id_aug_rate = list(map(float, id_aug_rate.split(',')))
    
    # organize image files by class
    class_to_images = defaultdict(list)
    for image_file in all_image_files:
        label, classname = get_label_and_classname(image_file, id2clsname)
        class_to_images[label].append(image_file)
    
    # invalid mask images
    invalid_mask_images = set()
    
    class_sample_counts = defaultdict(int)
    total_images = len(image_files)
    target_num = int(total_images * id_aug_rate[0])
    
    for option, times in zip(id_aug_options, id_aug_times):
        if times == 0:
            print(f'skip {option}')
            continue
        
        for _ in range(times):
            augmented_count = 0  
            
            if option == 1:
                background_type = 'inpaint'
                background_dir = inpainted_dir
                score_threshold = inpaint_thre
            elif option == 2:
                background_type = 'texture'
                background_dir = texture_dir
                score_threshold = None
            elif option == 3:
                background_type = 'noise'
                background_dir = None
                score_threshold = None
            else:
                continue
            
            used_backgrounds = set()
            tmp = image_files.copy()
            
            # augment images until we reach the target number
            while augmented_count < target_num:
                if not tmp:
                    print(f"warning: no more images to augment")
                    break
                
                remaining = target_num - augmented_count
                sample_files = random.sample(tmp, min(remaining, len(tmp)))
                
                # check if all remaining images are invalid
                all_invalid = all(image_file in invalid_mask_images for image_file in sample_files)
                if all_invalid:
                    print(f"no more valid images to augment")
                    break
                
                for image_file in sample_files:
                    if image_file in invalid_mask_images:
                        continue
                    
                    print(f'augmenting: {image_file}')
                    res = replace_background(
                        image_file, image_dir, mask_dir, background_dir, 
                        background_type, score_threshold, used_backgrounds, 
                        mask_thre, id2clsname
                    )
                    
                    if res is not None:
                        base_image_name = os.path.splitext(image_file)[0]
                        # timestamp = int(time.time() * 1000)
                        # random_suffix = random.randint(0, 1000)
                        augmented_image_path = os.path.join(
                            save_path, 
                            # f"{base_image_name}_aug_{option}_{timestamp}_{random_suffix}.png"
                            f"{base_image_name}_aug_{option}.png"
                        )
                        print(f"save augmented samples to {augmented_image_path}")
                        res.save(augmented_image_path)
                        
                        label, _ = get_label_and_classname(image_file, id2clsname)
                        class_sample_counts[label] += 1
                        augmented_count += 1  
                        tmp.remove(image_file)
                        
                        if augmented_count >= target_num:  
                            break
                    else:
                        print(f"failed to augment: {image_file}")
                        tmp.remove(image_file)
                        invalid_mask_images.add(image_file)
    
    # find the maximum number of samples for any class
    max_samples = max(class_sample_counts.values(), default=0)
    
    # copy existing samples to reach the maximum number of samples
    for label, images in class_to_images.items():
        original_images = images.copy()  
        while class_sample_counts[label] < max_samples and original_images:
            remaining = max_samples - class_sample_counts[label]
            sample_files = random.sample(original_images, min(remaining, len(original_images)))
            
            for image_file in sample_files:
                base_image_name = os.path.splitext(image_file)[0]
                timestamp = int(time.time() * 1000)
                random_suffix = random.randint(0, 1000)
                image_path = os.path.join(image_dir, image_file)
                repeated_image_path = os.path.join(
                    save_path, 
                    f"{base_image_name}_repeat_{timestamp}_{random_suffix}.png"
                )
                print(f"save copied samples to {repeated_image_path}")
                shutil.copy(image_path, repeated_image_path)
                
                class_sample_counts[label] += 1
                original_images.remove(image_file)  
                
                if class_sample_counts[label] >= max_samples:
                    break
    
    print('ID augmentation completed')
    print(f"{sum(class_sample_counts.values())} augmented samples are generated")
    print(f"augmented samples saved to {save_path}")


def replace_background(image_file, image_dir, mask_dir, background_dir=None, background_type='inpaint', score_threshold=None, used_backgrounds=set(), mask_thre=0.5, id2clsname=None):
    import clip_w_local
        
    model, _ = clip_w_local.load('ViT-B/16', device='cpu')
        
    base_image_name = os.path.splitext(image_file)[0]
    mask_file = next((f for f in os.listdir(mask_dir) if os.path.splitext(f)[0] == base_image_name), None)
    image_path = os.path.join(image_dir, image_file)
    
    # check if image file exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
        
    image = Image.open(image_path).convert('RGBA')
    current_class = base_image_name.split('_')[0]
    label, classname = get_label_and_classname(image_file, id2clsname)
    
    if mask_file is None:
        print(f"mask not founded: {image_file}")
        return None
        
    mask_path = os.path.join(mask_dir, mask_file)
    
    # check if mask file exists
    if not os.path.exists(mask_path):
        print(f"Mask file not found: {mask_path}")
        return None
        
    mask = Image.open(mask_path).convert('L')
    
    if area_ratio(mask) < mask_thre:
        print(f"mask area too small: {image_file}")
        return None
    
    for repeat in range(5):
        if background_type == 'inpaint':
            available_backgrounds = [f for f in os.listdir(background_dir) if f not in used_backgrounds]
            if not available_backgrounds:
                print(f"No available inpaint backgrounds for inpainting: {image_file}")
                return None
                
            background_file = random.choice(available_backgrounds)
            if os.path.splitext(background_file)[0] == base_image_name or background_file.split('_')[0] == current_class:
                continue
                
            if score_threshold is not None and parse_score(background_file) < score_threshold:
                continue
                
            background_path = os.path.join(background_dir, background_file)
            if not os.path.exists(background_path):
                print(f"Background file not found: {background_path}")
                continue
                
            background = Image.open(background_path).convert('RGBA')
            if is_white_or_black_background(background):
                continue
            
        elif background_type == 'texture':
            available_backgrounds = [f for f in os.listdir(background_dir) if f not in used_backgrounds]
            if not available_backgrounds:
                print(f"No available texture backgrounds for inpainting: {image_file}")
                return None
                
            background_file = random.choice(available_backgrounds) 
            if os.path.splitext(background_file)[0] == base_image_name or background_file.split('_')[0] == current_class:
                continue
            
            background_path = os.path.join(background_dir, background_file)
            if not os.path.exists(background_path):
                print(f"Background file not found: {background_path}")
                continue
                
            background = Image.open(background_path).convert('RGBA')
            
        elif background_type == 'noise':
            background = Image.fromarray(np.random.randint(0, 256, image.size + (4,), dtype=np.uint8))
            
        else:
            continue
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        if mask_np.ndim > 2:
            mask_np = mask_np[:, :, 0]
            
        background = background.resize(image.size)
        background_np = np.array(background)
        
        if mask_np.shape[:2] != image_np.shape[:2] or mask_np.shape[:2] != background_np.shape[:2]:
            print(f"Shape mismatch: mask {mask_np.shape}, image {image_np.shape}, background {background_np.shape}")
            continue
            
        mask_3d = mask_np[:, :, None]
        
        result_np = np.where(mask_3d > 0, image_np, background_np)
        result = Image.fromarray(result_np.astype(np.uint8)).convert('RGB')  # convert to RGB mode
    
        # convert the image to RGB mode
        transform = transforms.ToTensor()
        image_tensor = transform(image.convert('RGB'))
        result_tensor = transform(result)
    
        # ensure that the image tensors have the same dtype as the model
        image_tensor = image_tensor.type(model.dtype)
        result_tensor = result_tensor.type(model.dtype)
    
        # move the image tensor to the same device as the model
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        result_tensor = result_tensor.to(device)
    
        # Ensure that the tensors have the correct shape(batch_size, channels, height, width)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if len(result_tensor.shape) == 3:
            result_tensor = result_tensor.unsqueeze(0)
    
        # calculate the similarity between the original and augmented images
        original_features, _ = model.visual(image_tensor)
        augmented_features, _ = model.visual(result_tensor)
        original_features = original_features / original_features.norm(dim=-1, keepdim=True)
        augmented_features = augmented_features / augmented_features.norm(dim=-1, keepdim=True)
        similarity = (original_features * augmented_features).sum(dim=-1)
            
        # set a threshold for similarity, which can be adjusted based on experiments
        threshold = 0.7  
    
        # filter out samples with low similarity (which are too difficult)
        if similarity < threshold:
            print(f"aug samples are too difficult: {image_file}, similarity: {similarity.item()}")
            continue
            
        used_backgrounds.add(background_file)
        return result
        
    return None

# function to get the label and classname from the image file name
def get_label_and_classname(image_file, id2clsname):
       
    # get the base name of the image file without extension
    base_image_name = os.path.splitext(os.path.basename(image_file))[0]
        
    # for the original file: try to extract the ID
    if base_image_name.startswith('n') and '_' in base_image_name:
        imageid = base_image_name.split('_')[0]  # 'n01440764' from  'n01440764_3236' 
        
    # for the augmented file: check if it contains 'aug' or 'repeat'
    elif '_aug_' in base_image_name or '_repeat_' in base_image_name:
        parts = base_image_name.split('_')
        if parts[0].startswith('n'):
            imageid = parts[0]  
        else:
            print(f"Warning: Could not extract proper ID from augmented image: {image_file}")
            # try to find the ID in the original file name
            for id in id2clsname.keys():
                if id in base_image_name:
                    imageid = id
                    break
            else:
                # return -1 if no valid ID is found
                print(f"Error: No valid ImageNet ID found in filename: {image_file}")
                return -1, "unknown"
    else:
        # otherwise, assume the ID is the first part of the filename
        parts = base_image_name.split('_')
        imageid = parts[0]
            
    # look up the ID in the class mapping
    if imageid in id2clsname:
        label, classname = id2clsname[imageid]
        return label, classname
    else:
        print(f"Warning: ID '{imageid}' from file '{image_file}' not found in class mapping")
        return -1, "unknown"

# function to calculate the area ratio of the mask
def area_ratio(mask):
    mask_np = np.array(mask)
    non_zero_count = np.count_nonzero(mask_np)
    total_count = mask_np.size
    return non_zero_count / total_count

# function to parse the score from the filename
def parse_score(filename):
    try:
        return float(filename.split('_')[-1].split('.')[0])
    except ValueError:
        return 0.0

# function to check if the image is white or black background
def is_white_or_black_background(image):
    image_np = np.array(image)
    is_white = np.all(image_np[:, :, :3] == 255)
    is_black = np.all(image_np[:, :, :3] == 0)
    return is_white or is_black

def load_imagenet_classnames(classnames_path):

    id2clsname = {}
    class_idx = 0
        
    with open(classnames_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                image_id = parts[0]  
                classname = parts[1]  
                id2clsname[image_id] = [class_idx, classname]
                class_idx += 1
        
    return id2clsname

def main(args):
    """
    主函数，根据命令行参数应用ID增强
    """
    print("Starting ID augmentation with following parameters:")
    print(f"Train root: {args.train_root}")
    print(f"Mask threshold: {args.mask_thre}")
    print(f"Shots: {args.shots}")
    print(f"Seed: {args.seed}")
    print(f"ID augmentation options: {args.id_aug_options}")
    print(f"ID augmentation times: {args.id_aug_times}")
    print(f"ID augmentation rate: {args.id_aug_rate}")

    apply_id_aug(
        train_root=args.train_root,
        mask_thre=args.mask_thre,
        shots=args.shots,
        seed=args.seed,
        id_aug_options=args.id_aug_options,
        id_aug_times=args.id_aug_times,
        id_aug_rate=args.id_aug_rate
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ID Augmentation for OOD Detection")
    
    parser.add_argument(
        "--train_root", type=str, default="ImageNet_1shot_seed1",
        help="root directory name for training data, e.g., 'ImageNet_1shot_seed1'"
    )
    
    parser.add_argument(
        "--mask_thre", type=float, default=0.5,
        help="threshold for mask area ratio"
    )
    
    parser.add_argument(
        "--shots", type=int, default=1,
        help="number of shots for few-shot learning"
    )
    
    parser.add_argument(
        "--seed", type=int, default=1,
        help="random seed for reproducibility"
    )
    
    parser.add_argument(
        "--id_aug_options", type=str, default="1,2,3",
        help="ID augmentation options, e.g., '1,2,3' for inpaint, texture, and noise"
    )
    
    parser.add_argument(
        "--id_aug_times", type=str, default="1,1,1",
        help="number of times for each augmentation option, e.g., '1,1,1'"
    )
    
    parser.add_argument(
        "--id_aug_rate", type=str, default="0.5,0.5,0.5",
        help="augmentation rate for each option, e.g., '0.5,0.5,0.5'"
    )

    args = parser.parse_args()
    
    main(args)