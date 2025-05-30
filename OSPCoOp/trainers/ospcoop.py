import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from dassl.data.datasets import Datum
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip_w_local import clip
import numpy as np
from tqdm import tqdm
from torchvision import datasets
from utils.detection_util import get_and_print_results
from dassl.data.transforms import build_transform
from torch.utils.data import Dataset as TorchDataset
from utils.training_util import build_data_loader
from .custom_clip import load_clip_to_cpu, TextEncoder, PromptLearner, CustomCLIP


def entropy_select_topk(p, bs, label, num_of_local_feature, mask):
    if num_of_local_feature == 196:
        patch_size = 16
    else:
        patch_size = 32
    mask_patches = mask.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patch_counts = (mask_patches == 255).sum(dim=(3, 4))
    patch_counts = patch_counts.view(bs, -1)
    patch_counts = patch_counts > 100
    patch_counts[label == p.shape[1]] = False

    patch_counts = patch_counts.view(-1)
    label_repeat = label.repeat_interleave(num_of_local_feature)
    p = F.softmax(p, dim=-1)
    selected_p = p[~patch_counts]

    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(selected_p * torch.log(selected_p + 1e-5), 1))


@TRAINER_REGISTRY.register()
class OSPCoOp(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.OSPCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        cfg.num_classes = len(classnames)
        self.DATA_ROOT = cfg.DATA_ROOT[0]
        self.num_classes = len(classnames)
        self.train_root = self.cfg.train_root
        self.loss1 = cfg.loss1
        self.loss2 = cfg.loss2
        self.glmcm_local_weight = cfg.glmcm_local_weight
        self.T = cfg.T
        self.eval_bg = cfg.eval_bg
        self.ood_aug = cfg.ood_aug
        self.id_aug_dir = cfg.id_aug_dir
        self.eval_freq = cfg.eval_freq
        self.start_eval_epoch = cfg.start_eval_epoch
        self.inpaint_thre = cfg.inpaint_thre
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.OSPCOOP.PREC == "fp32" or cfg.TRAINER.OSPCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.OSPCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        self.ood_loader()
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, mask = self.parse_batch_train(batch)

        output, output_local = self.model(image)

        valid_indices = (label >= 0) & (label <= output.shape[1] - 1)

        filtered_output = output[valid_indices]
        filtered_gt = label[valid_indices]

        loss_id = F.cross_entropy(filtered_output, filtered_gt)

        valid_indices_ood = (label == output.shape[1])
        filtered_output_ood = output[valid_indices_ood]
        filtered_output_ood = F.softmax(filtered_output_ood, dim=-1)

        loss_ood = torch.mean(torch.sum(filtered_output_ood * torch.log(filtered_output_ood + 1e-5), 1))

        batch_size, num_of_local_feature = output_local.shape[0], output_local.shape[1]

        output_local = output_local.view(batch_size * num_of_local_feature, -1)
        loss_en = - entropy_select_topk(output_local, batch_size, label, num_of_local_feature, mask)

        if filtered_output.shape[0] == 0:
            loss = self.loss1 * loss_en + self.loss2 * loss_ood
            self.model_backward_and_update(loss)
            loss_summary = {
                "loss": loss.item(),
                "loss_ood": loss_ood.item(),
                "loss_en": loss_en.item(),
            }
        elif filtered_output_ood.shape[0] == 0:
            loss = loss_id + self.loss1 * loss_en
            self.model_backward_and_update(loss)
            loss_summary = {
                "loss": loss.item(),
                "loss_id": loss_id.item(),
                "loss_en": loss_en.item(),
                "acc": compute_accuracy(filtered_output, filtered_gt)[0].item(),
            }
        else:
            loss = loss_id + self.loss1 * loss_en + self.loss2 * loss_ood
            self.model_backward_and_update(loss)
            loss_summary = {
                "loss": loss.item(),
                'loss_id': loss_id.item(),
                "loss_ood": loss_ood.item(),
                "loss_en": loss_en.item(),
                "acc": compute_accuracy(filtered_output, filtered_gt)[0].item(),

            }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        mask = batch["mask"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, mask

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def train(self):  # , start_epoch, max_epoch):
        """Generic training loops."""
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            if (self.epoch + 1) > self.start_eval_epoch:
                if (self.epoch + 1) % self.eval_freq == 0:
                    mcm_aurod, glmcm_auroc = self.eval_ood()
                    self.after_epoch(glmcm_auroc)

                elif (self.epoch + 1) == self.max_epoch:
                    mcm_aurod, glmcm_auroc = self.eval_ood()
                    self.after_epoch(glmcm_auroc)
        self.load_model(directory=self.output_dir)
        _, _ = self.eval_ood()

    def after_epoch(self, auroc):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST

        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        curr_result = auroc
        is_best = curr_result > self.best_result
        if is_best:
            self.best_result = curr_result
            self.save_model(
                self.epoch,
                self.output_dir,
                val_result=curr_result,
                model_name="model-best.pth.tar"
            )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test_ood(self, data_loader, T):
        """Test-time OOD detection pipeline."""
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

        glmcm_score = []
        mcm_score = []
        glmcm_score_entropy = []
        for batch_idx, (images, labels, *id_flag) in enumerate(tqdm(data_loader)):
            images = images.cuda()
            output, output_local = self.model_inference(images)

            output /= 100.0
            output_local /= 100.0
            smax_global = to_np(F.softmax(output / T, dim=-1))
            smax_local = to_np(F.softmax(output_local / T, dim=-1))
            mcm_global_score = -np.max(smax_global, axis=1)
            local_score = -np.max(smax_local, axis=(1, 2))

            mcm_score.append(mcm_global_score)
            glmcm_score.append(local_score)

        return concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(glmcm_score)[
                                                                    :len(data_loader.dataset)].copy()

    @torch.no_grad()
    def eval_ood(self):

        _, preprocess = clip.load("ViT-B/16", device='cpu')

        id_data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(self.DATA_ROOT['ID'], transform=preprocess),
            batch_size=512, shuffle=False, num_workers = 4, pin_memory = True)

        auroc_list_mcm, aupr_list_mcm, fpr_list_mcm = [], [], []
        auroc_list_gl, aupr_list_gl, fpr_list_gl = [], [], []

        in_score_mcm, in_score_gl = self.test_ood(id_data_loader, T=self.T)

        if self.eval_bg:
            out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture','ImageNet-Bg','ImageNet-Bg(S)']
        else:
            out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture']
        in_score_gl = in_score_gl * self.glmcm_local_weight + in_score_mcm

        for out_dataset in out_datasets:
            print(f"Evaluting OOD dataset {out_dataset}")
            ood_loader = self.set_ood_loader_ImageNet(out_dataset, preprocess)
            out_score_mcm, out_score_gl = self.test_ood(ood_loader, T=self.T)

            out_score_gl = out_score_gl * self.glmcm_local_weight + out_score_mcm
            print("MCM score")
            get_and_print_results(in_score_mcm, out_score_mcm,
                                  auroc_list_mcm, aupr_list_mcm, fpr_list_mcm)

            print("GL-MCM score")
            get_and_print_results(in_score_gl, out_score_gl,
                                  auroc_list_gl, aupr_list_gl, fpr_list_gl)

        print("MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_mcm), np.mean(auroc_list_mcm),
                                                          np.mean(aupr_list_mcm)))
        print("GL-MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_gl), np.mean(auroc_list_gl),
                                                             np.mean(aupr_list_gl)))

        return np.mean(auroc_list_mcm), np.mean(auroc_list_gl)

    def set_ood_loader_ImageNet(self, out_dataset, preprocess):
        testsetout = datasets.ImageFolder(root=self.DATA_ROOT['OOD'][out_dataset],
                                          transform=preprocess)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=512,
                                                    shuffle=False, num_workers=4)
        return testloaderOut

    @torch.no_grad()
    def ood_loader(self):
        self.inpainted_dir = './data/' + self.train_root + '/score_inpaint/'
        self.texture_dir = './data/' + self.train_root + '/texture/inpaint/top/'
        trainset = self.dm.train_loader_x.dataset
        tmp_data_source = []

        files = []
        self.id2clsname = {}
        for item in trainset.data_source:
            self.id2clsname[item.impath.split('/')[-1].split('_')[0]] = [item.label, item.classname]
        # load background data 
        for filename in os.listdir(self.inpainted_dir):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                impath = os.path.join(self.inpainted_dir, filename)
                scores = float(os.path.splitext(filename)[0].split('_')[2])
                if scores > self.inpaint_thre:
                    classname = 'background'
                    label = self.num_classes
                    tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=2)) # Domain 2: Background
                    files.append(filename.split('_')[0] + '_' + filename.split('_')[1])
        # load OOD Aug data 
        if self.ood_aug:
            image_folder = self.texture_dir
            for filename in os.listdir(image_folder):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                    impath = os.path.join(image_folder, filename)
                    classname = 'background'
                    label = self.num_classes
                    tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=2))
        # load few-shot ID data 
        image_folder_raw = './data/' + self.train_root + '/raw/'
        for filename in os.listdir(image_folder_raw):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                impath = os.path.join(image_folder_raw, filename)
                imageid = filename.split('_')[0]
                classname = self.id2clsname[imageid][1]
                label = self.id2clsname[imageid][0]
                if filename.split('.')[0] in files:
                    tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=1)) # Domain 1: ID data with mask
                else:
                    tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=0)) # Domain 0: ID data without mask
        # load id_aug data
        if self.id_aug_dir:
            for filename in os.listdir(self.id_aug_dir):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                    impath = os.path.join(self.id_aug_dir, filename)
                    imageid = filename.split('_')[0]
                    classname = self.id2clsname[imageid][1]
                    label = self.id2clsname[imageid][0]
                    if filename.split('_')[0] + '_' + filename.split('_')[1] in files:
                        tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=1))
                    else:
                        tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=0))

        trainset.data_source = tmp_data_source
        tfm_train = build_transform(self.cfg, is_train=True)
        self.train_loader_x = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=trainset,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
        )
