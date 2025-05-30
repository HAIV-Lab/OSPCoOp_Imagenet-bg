import argparse
import torch
from datetime import datetime
from dassl.utils import setup_logger, set_random_seed
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from root_config import get_root
import contextlib
import io
import datasets.imagenet
import trainers.ospcoop


def print_args(args):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))


def reset_cfg(cfg, args):
    cfg.DATA_ROOT = get_root()
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    else:
        now = datetime.now()
        formatted_time = now.strftime('%Y%m%d%H%M%S')
        cfg.OUTPUT_DIR = './runs/' + formatted_time

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    cfg.loss1 = args.loss1
    cfg.loss2 = args.loss2
    cfg.glmcm_local_weight = args.glmcm_local_weight
    cfg.T = args.T
    cfg.eval_bg = args.eval_bg
    cfg.id_aug_dir = args.id_aug_dir
    cfg.mask_thre_for_resize_crop = args.mask_thre_for_resize_crop
    cfg.ood_aug = args.ood_aug

    cfg.inpaint_thre = args.inpaint_thre
    cfg.mask_thre = args.mask_thre
    cfg.shots = args.shots
    cfg.train_root = 'ImageNet_' + str(args.shots) + 'shot' + '_seed' + str(args.seed)

    cfg.eval_freq = args.eval_freq
    cfg.start_eval_epoch = args.start_eval_epoch


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    cfg.IMAGE_SIZE = 224
    cfg.TRAINER.OSPCOOP = CN()
    cfg.TRAINER.OSPCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.OSPCOOP.CSC = False  # class-specific context
    cfg.TRAINER.OSPCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.OSPCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.OSPCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()

    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    #cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args)
    with contextlib.redirect_stdout(io.StringIO()):
        trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.eval_ood()

    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, default="", help="output directory")

    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )

    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/OSPCoOp/vit_b16_ep50.yaml", help="path to config file"
    )

    parser.add_argument(
        "--ood-aug",
        type=bool,
        default=True,
        help="whether to apply ood aug (True/False)"
    )
    
    parser.add_argument(
        "--eval-bg",
        type=bool,
        default=False,
        help="whether to eval ImageNet-Bg (True/False)"
    )
    
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/imagenet.yaml",
        help="path to config file for dataset setup",
    )
    
    parser.add_argument("--trainer", type=str, default="OSPCoOp", help="name of trainer")
    
    parser.add_argument('--shots', type=int, default=16,
                        help='weight for regulization loss')
                                                                   
    parser.add_argument('--loss1', type=float, default=1.5,
                        help='weight for regulization loss')

    parser.add_argument('--loss2', type=float, default=0.5,
                        help='weight for regulization loss')
                        
    parser.add_argument('--inpaint_thre', type=float, default=5)

    parser.add_argument('--mask_thre', type=float, default=50)

    parser.add_argument(
        "--id_aug_dir", type=str, default='', help="load ID aug data."
    )

    parser.add_argument('--mask_thre_for_resize_crop', type=float, default=0.2,
                        help='mask thre for randomresize crop, used with id aug')
                        
    parser.add_argument('--glmcm_local_weight', type=float, default=0.5,
                        help='weight for glmcm')
                        
    parser.add_argument('--T', type=float, default=1,
                        help='temperature for softmax') 

    parser.add_argument(
        "--eval_freq", type=int, default=1, help="load model weights at this epoch for evaluation"
    )
    
    parser.add_argument(
        "--start_eval_epoch", type=int, default=5, help="load model weights at this epoch for evaluation"
    )
    
    parser.add_argument("--eval_only", default=False, action="store_true", help="evaluation only")

    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    

    args = parser.parse_args()
    
    main(args)

