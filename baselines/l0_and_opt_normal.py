import sys
sys.path.append("..")
import torch
import torchvision
from torch.autograd.functional import jacobian
from transforms import transform_wb, get_transform_params, convert2NetworkWB, get_transformed_images
from utils import run_predictions
from GTSRB.GTSRBNet import GTSRBNet
from tqdm import tqdm
import pickle
import cv2
import numpy as np
import random
from torchvision.utils import save_image
import os
import argparse
from opt_normal import attack_targeted
from prettytable import PrettyTable
import pickle
import seed


class Logger:
    def __init__(self, args):
        self.args = args

        self.log_dict = {
            'victim_label': args.victim_label,
            'target_label': args.target_label,
            'args': args,
            'round_results': {}
        }

    def update(self, current_round, transform_robustness, query_count, mask, adv_img):
        self.log_dict['round_results'][current_round] = {
            'query_count': query_count,
            'transform_robustness': transform_robustness,
            'mask': mask,
            'adv_img': adv_img
        }

    def save(self):
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, 'results_victim_label={}_target_label={}.pkl'.format(
                self.args.victim_label, self.args.target_label)), 'wb') as f:
            pickle.dump(self.log_dict, f)

        rounds = list(self.log_dict['round_results'].keys())
        rounds.sort()
        if len(rounds) > 0:
            last_round = rounds[-1]
            print("Final query count:", self.log_dict['round_results'][last_round]['query_count'])
            print("Final transform_robustness:", self.log_dict['round_results'][last_round]['transform_robustness'])
            print("Final mask_size:", torch.sum(self.log_dict['round_results'][last_round]['mask']).item() / 3)
            save_image(self.log_dict['round_results'][last_round]['adv_img'], os.path.join(self.args.log_dir, 'results_victim_label={}_target_label={}.png'.format(self.args.victim_label, self.args.target_label)))
            save_image(self.log_dict['round_results'][last_round]['mask'], os.path.join(self.args.log_dir, 'results_victim_label={}_target_label={}_mask.png'.format(self.args.victim_label, self.args.target_label)))


def compute_transform_robustness(img, delta, mask, model, xforms, xforms_pt_file, model_input_size, target_label):
    xform_imgs = get_transformed_images(img.detach().cpu(), mask, xforms, 1.0, delta.detach().cpu(),
                                        xforms_pt_file,
                                        net_size=model_input_size)
    neg_tr, qc = run_predictions(model, xform_imgs, -1, target_label)
    return 1 - neg_tr, qc


def main(args):
    print('-----------------------------------------------------------------------------------')
    print('Running baseline: {}'.format(__file__.split('.')[0]))
    print('-----------------------------------------------------------------------------------')
    assert args.model == 'GTSRB'
    args_table = PrettyTable(['Argument', 'Value'])
    for arg in vars(args):
        args_table.add_row([arg, getattr(args, arg)])
    print(args_table)
    logger = Logger(args)

    # Load victim img.
    img = cv2.imread(args.victim_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_large = np.array(img, dtype=np.float32) / 255.0
    img_large = torch.from_numpy(img_large).cuda().permute((2, 0, 1)).cuda()
    img = cv2.resize(img, (32, 32))
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).cuda().permute((2, 0, 1)).cuda()
    img.requires_grad = True

    # Load target img.
    tar_img = cv2.imread(args.target_img_path)
    tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
    tar_img = cv2.resize(tar_img, (32, 32))
    tar_img = np.array(tar_img, dtype=np.float32) / 255.0
    tar_img = torch.from_numpy(tar_img).cuda().permute((2, 0, 1)).cuda()

    # Load starting mask.
    mask = cv2.imread(args.initial_mask_path)
    mask = np.where(mask > 128, 255, 0)
    mask = torch.from_numpy(mask).permute(2, 0, 1).cuda() / 255.0

    # Load transforms.
    xforms = get_transform_params(args.num_xforms, args.model, baseline = True)

    # Load net.
    net = GTSRBNet()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
    net.eval()
    model = net.module if torch.cuda.is_available() else net
    checkpoint = torch.load('../GTSRB/checkpoint_us.tar')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Attack.
    query_count = 0
    print('Initializing...')
    opt_theta_initializer = None
    opt_lbd_initializer = None
    for rounds in range(args.max_rounds):
        print('Beginning round: {}'.format(rounds))
        print('Stage 1: Opt-Attack.')

        # Gradient estimation.
        print('Searching for adversarial example...')
        adv_img, opt_theta_initializer, qc, _, final_avg_grad, opt_lbd_initializer = attack_targeted(model,
                                                                                       [(tar_img.detach().cpu(),
                                                                                         args.target_label)],
                                                                                       img.detach().cpu(),
                                                                                       args.victim_label, args.target_label,
                                                                                       mask.cpu(), opt_theta_initializer, opt_lbd_initializer)
        query_count += qc
        adv_img = adv_img.detach().cuda()

        if opt_lbd_initializer < 0 or run_predictions(model, [adv_img], -1, args.target_label)[0] != 0:
            print('Opt could not find an adversarial example. End.')
            query_count += 1
            break

        query_count += 1
        transform_robustness, qc = compute_transform_robustness(adv_img, torch.zeros_like(adv_img), mask, model, xforms,
                                                  args.xforms_pt_file, args.model_input_size, args.target_label)
        query_count += qc
        print('Found adversarial example | transform_robustness: {}%'.format(transform_robustness * 100))

        delta_np = (adv_img - img).permute(1, 2, 0).detach().cpu().numpy()
        delta_np_large = cv2.resize(delta_np, (244, 244))
        delta_large_torch = torch.from_numpy(delta_np_large).permute(2, 0, 1).to(img.device)
        adv_img_large = img_large + delta_large_torch
        logger.update(rounds, transform_robustness, query_count, mask, adv_img_large)

        if final_avg_grad is None:
            print("Opt could not find a gradient direction. End")
            break

        final_avg_grad = final_avg_grad.cuda()

        print('Stage 2: Mask reduction.')
        pert = adv_img - img
        final_avg_grad[torch.isnan(final_avg_grad)] = 0
        final_avg_grad = mask * final_avg_grad * pert
        pixelwise_avg_grads = torch.sum(torch.abs(final_avg_grad), dim=0)

        # Find minimum gradient patches and remove them.
        for _ in range(args.patches_per_round):
            patch_removal_size = args.patch_removal_size
            patch_removal_interval = args.patch_removal_interval
            min_patch_grad = 99999999999999999
            min_patch_grad_idx = None
            for i in range(0, pixelwise_avg_grads.shape[0] - patch_removal_size + 1, patch_removal_interval):
                for j in range(0, pixelwise_avg_grads.shape[1] - patch_removal_size + 1, patch_removal_interval):
                    patch_grad = pixelwise_avg_grads[i:i + patch_removal_size, j:j + patch_removal_size].sum()
                    if mask[0, i:i + patch_removal_size, j:j + patch_removal_size].sum() > 0:
                        patch_grad = patch_grad / mask[0, i:i + patch_removal_size, j:j + patch_removal_size].sum()
                        if patch_grad.item() < min_patch_grad:
                            min_patch_grad = patch_grad.item()
                            min_patch_grad_idx = (i, j)
            if min_patch_grad_idx is None:
                continue
            i, j = min_patch_grad_idx
            mask[0, i:i + patch_removal_size, j:j + patch_removal_size] = 0
            mask[1, i:i + patch_removal_size, j:j + patch_removal_size] = 0
            mask[2, i:i + patch_removal_size, j:j + patch_removal_size] = 0
            print("Removed patch: {}".format((i, j)))
        print('-----------------------------------------------------------------------------------')
        if torch.sum(mask) == 0:
            break
    logger.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_label', type=int, default=1)
    parser.add_argument('--victim_label', type=int, default=14)
    parser.add_argument('--num_xforms', type=int, default=100)
    parser.add_argument('--xforms_pt_file', type=str, default='../inputs/GTSRB/Points/14.csv')
    parser.add_argument('--model_input_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='GTSRB')
    parser.add_argument('--patch_removal_size', type=int, default=4)
    parser.add_argument('--patch_removal_interval', type=int, default=1)
    parser.add_argument('--patches_per_round', type=int, default=4)
    parser.add_argument('--max_rounds', type=int, default=999999999)
    parser.add_argument('--log_dir', type=str, default='../logs/l0_and_opt_normal')
    parser.add_argument('--victim_img_path', type=str, default='../inputs/GTSRB/images/14.png')
    parser.add_argument('--target_img_path', type=str, default='../inputs/GTSRB/images/1.png')
    parser.add_argument('--initial_mask_path', type=str, default='../plain_masks/mask.png')

    main(parser.parse_args())
