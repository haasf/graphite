import time
import random 
import numpy as np
import cv2
import sys
from utils import run_predictions, grayDim
import torch 
import torchvision.transforms as transforms
from torchvision import models
from GTSRB.GTSRBDataset import GTSRBDataset
from GTSRB.GTSRBNet import GTSRBNet
from cifar.wideresnet import WideResNet
from boost import boost
from transforms import get_transform_params, get_transformed_images, convert2Network, add_noise
from generate_mask import generate_mask
from shutil import copyfile
import parsearguments
from scipy.cluster.vq import vq

from MNIST.mnist_net import MnistNet
import random
def attack_network(model, img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta= 1, num_xforms_mask = 100, num_xforms_boost = 1000, net_size = 32, noise_size = 32, model_type = 'GTSRB', joint_iters = 1, image_id='',return_type = 'Stats'):
    import getpass
    username = getpass.getuser()
    args = parsearguments.getarguments()
    coarse_mode = args.coarse_mode
    

    if image_id != '':
        out_str_base = username + "_" + image_id + '_' + str(lbl_v) + '_' + str(lbl_t) + '_' + str(reduceerror)  + '_' + str(coarseerror) + '_' +  str(coarse_mode) + '_' + scorefile + "_" ;
    else:
        out_str_base = username + "_" + str(lbl_v) + '_' + str(lbl_t) + '_' + str(reduceerror)  + '_' + str(coarseerror) + '_' +  str(coarse_mode) + '_' + scorefile + "_" ;
    out_str_heat = username + "_" + str(lbl_v) + '_' + str(lbl_t)
    output_base = args.out_path
    if model_type != 'GTSRB':
        output_base += model_type + '/'


    print("Attacking network. model_type = ", model_type, " image_id = ", image_id)
    cv2Color = cv2.COLOR_BGR2RGB
    numColors = 3
    if(model_type == 'MNIST'):
        cv2Color = cv2.COLOR_BGR2GRAY
        numColors = 1
    
    img = cv2.imread(img_v)
    img = cv2.cvtColor(img, cv2Color)
    img = np.array(img, dtype=np.float32) / 255.0

    img_small = cv2.resize(img, (noise_size, noise_size))

    img = grayDim(img)
    img_small = grayDim(img_small)

    img_torch = torch.from_numpy(img).permute(2, 0, 1)
    img_small_torch = torch.from_numpy(img_small).permute(2, 0, 1)
    
    assert args.square_x is None and args.square_y is None or args.square_x is not None and args.square_y is not None

    mask_file = None
    if mask is not None:
        mask_file = mask
        mask_full = cv2.imread(mask)
        if(numColors == 1):
            mask_full = cv2.cvtColor(mask_full,cv2Color)

        if mask_full.shape[0] != noise_size or mask_full.shape[1] != noise_size:
            mask = cv2.resize(mask_full, (noise_size, noise_size))

        else:
            mask = mask_full
            
        mask = grayDim(mask)
        mask = np.where(mask > 128, 255, 0) 
        mask = torch.from_numpy(mask).permute(2, 0, 1) / 255.0

    if img_t is not None:
        tar_large = cv2.imread(img_t)
        tar = cv2.resize(tar_large, (noise_size, noise_size))
        tar = cv2.cvtColor(tar, cv2Color)
        tar = np.array(tar, dtype=np.float32) / 255.0

        
        tar_large = cv2.cvtColor(tar_large, cv2Color)
        tar_large = np.array(tar_large, dtype=np.float32) / 255.0

        tar = grayDim(tar)
        tar_large = grayDim(tar_large)

        tar = torch.from_numpy(tar).permute(2, 0, 1)
        tar_large = torch.from_numpy(tar_large).permute(2, 0, 1).float()


    prior_mask = mask.clone()
    next_theta = None
    prior_attack = tar.clone()
    prior_attack_large = tar_large.clone()
    total_mask_query_ct = 0
    total_boost_query_ct = 0

    # patch_size = 4
    patch_size = noise_size // 8
    # patch_size = 1
    for i in range(joint_iters):
        if args.square_x is not None:
            assert joint_iters == 1
            mask_query_count = 0
            mask_out = np.zeros((noise_size, noise_size, numColors))
            square_x = int(args.square_x)
            square_y = int(args.square_y)
            square_size = int(args.square_size)
            mask_out[square_y:square_y+square_size, square_x:square_x+square_size, :] = np.ones((square_size, square_size, numColors))
            mask_out = torch.from_numpy(mask_out).permute(2, 0, 1).float()
            best_theta = (tar - img_small_torch) * mask_out
            init = add_noise(img_torch, mask_out, 1.0, best_theta)
            nbits = square_size ** 2
            xforms = get_transform_params(num_xforms_mask, model_type)
            xform_imgs = get_transformed_images(img_torch, mask_out, xforms, 1.0, best_theta, pt_file, net_size = net_size, model_type = model_type)
            success_rate, query_ct = run_predictions(model, xform_imgs, lbl_v, lbl_t)
            tr_score = 1 - success_rate
            
        else:
            init, this_mask_query_count, mask_out, nbits, tr_score = \
                          generate_mask(model, img_small_torch, img_torch, lbl_v, prior_mask, prior_attack, prior_attack_large, lbl_t, pt_file, scorefile,
                                        heatmap, coarseerror, reduceerror, num_xforms = num_xforms_mask,
                                        net_size = net_size, model_type = model_type, patch_size = patch_size,
                                        heatmap_file = args.heatmap_file, heatmap_out_path = output_base + 'heatmaps/' + out_str_heat + '_heatmap.pkl', max_mask_size = args.max_mask_size, init_theta = next_theta,numColors = numColors)
            total_mask_query_ct += this_mask_query_count
        nbits = int(nbits)

        mask_out_np = mask_out.permute(1, 2, 0).numpy()
        cv2.imwrite(output_base + 'masks/' + out_str_base + '_' + str(nbits) + '_bits_' + str(tr_score) + '_tr_' + str(i) + '_joint_iter.png', mask_out_np * 255)

        init_np = init.permute(1, 2, 0).numpy()
        init_np = cv2.cvtColor(init_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_base + 'inits/' + out_str_base + '_' + str(nbits) + '_bits_' + str(tr_score) + '_tr_' + str(i) + '_joint_iter.png', init_np * 255)

        

        if nbits != 0:
            adversarial, tr_score, perturb, this_boost_query_count, next_theta, adv_small = boost(model, img_torch, lbl_v, mask_out, prior_attack, lbl_t, beta = beta, iterations = 1000, pt_file = pt_file, num_xforms = num_xforms_boost, net_size = net_size, model_type = model_type, square_mask = args.square_x is not None, early_boost_exit = args.early_boost_exit, init_theta = next_theta)
            total_boost_query_ct += this_boost_query_count
        else:  
            adversarial = img_torch
            tr_score = 1.0
            # could calculate TR from run_predictions
            # print("Num bits = 0, quitting")
            # quit()
            
        if return_type == 'Image':
            print("nbits = ", nbits)
            return adversarial, nbits #, tr_score
        

        prior_mask = mask_out.clone()
        prior_attack_large = adversarial.clone()
        prior_attack = torch.from_numpy(grayDim(cv2.resize(prior_attack_large.clone().permute(1, 2, 0).numpy(), (noise_size, noise_size)))).permute(2, 0, 1)

        # return adversarial (torch img)
        adversarial_np = adversarial.permute(1, 2, 0).numpy()

        adversarial_np = cv2.cvtColor(adversarial_np, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_base + 'boosted/' + out_str_base + '_' + str(nbits) + '_bits_' + str(tr_score) + '_tr_' + str(i) + '_joint_iter.png', adversarial_np * 255)

        if nbits != 0:
            perturb_np = perturb.permute(1, 2, 0).numpy()
            perturb_np = cv2.cvtColor(perturb_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_base + 'perturbations/' + out_str_base + '_' + str(nbits) + 'bits_' + str(tr_score) + '_tr_' + str(i) + '_joint_iter.png', perturb_np * 255)

    print("-------------------------------")
    print("Attack Completed.")
    if args.num_test_xforms > -1:
        xforms = get_transform_params(args.num_test_xforms, model_type)
        xform_imgs = get_transformed_images(adversarial, mask_out, xforms, 0.0, torch.zeros(mask_out.size()), pt_file, net_size = net_size, model_type = model_type)
        success_rate, query_ct = run_predictions(model, xform_imgs, lbl_v, lbl_t)
        print("Final transform_robustness (" + str(args.num_test_xforms) + " transforms):", 1.0 - success_rate)


    with open("modelTraining/MNIST/training.txt", mode = "a") as f: 
        f.write("\nIMAGE_ID: " + image_id)
        f.write(f"\nFinal transform_robustness: {tr_score}")
        f.write(f"\nFinal number of pixels: {nbits}") 
        f.write(f"\nFinal number of queries: {total_mask_query_ct + total_boost_query_ct} ") 

    print("Final transform_robustness:", tr_score)
    print("Final number of pixels:", nbits)
    print("Final number of queries:", total_mask_query_ct + total_boost_query_ct)
    #print("Final number of queries:", mask_query_count + boost_query_count)
    return tr_score, nbits, total_mask_query_ct + total_boost_query_ct


def attack_GTSRB(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta= 1, num_xforms_mask = 100, num_xforms_boost = 1000, net_size = 32, noise_size = 32, joint_iters = 1):
    net = GTSRBNet()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        
    net.eval()

    model = net.module if torch.cuda.is_available() else net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('GTSRB/checkpoint_us.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    attack_network(model, img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta, num_xforms_mask, num_xforms_boost, net_size, noise_size, model_type = 'GTSRB', joint_iters = joint_iters)


def attack_CIFAR(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta= 1, num_xforms_mask = 100, num_xforms_boost = 1000, net_size = 32, noise_size = 32, joint_iters = 1, image_id = ''):
    net = WideResNet(num_classes=10)
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        
    net.eval()

    model = net.module if torch.cuda.is_available() else net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('cifar/epoch-39.ckpt', map_location=device)
    model.load_state_dict(checkpoint)

    return attack_network(model, img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta, num_xforms_mask, num_xforms_boost, net_size, noise_size, model_type = 'CIFAR', joint_iters = joint_iters, image_id = image_id)


def attack_MNIST(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta= 1, num_xforms_mask = 100, num_xforms_boost = 1000, net_size = 32, noise_size = 32, joint_iters = 1, image_id = ''):
    net = MnistNet()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
    
    net.eval()

    model = net.module if torch.cuda.is_available() else net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('MNIST/epoch-39-MNIST.ckpt', map_location=device)
    
    model.load_state_dict(checkpoint)

    return attack_network(model, img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta, num_xforms_mask, num_xforms_boost, net_size, noise_size, model_type = 'MNIST', joint_iters = joint_iters, image_id = image_id)
if __name__ == '__main__':
    network = 'MNIST'
    # runType = 'PrelimStats'
    runType = "just one run"
    args = parsearguments.getarguments()
    network = args.network
    img_v = args.img_v
    img_t = args.img_t
    mask = args.mask
    lbl_v = args.lbl_v
    lbl_t = args.lbl_t
    pt_file = args.pt_file
    scorefile = args.scorefile
    heatmap = args.heatmap
    num_xforms_boost = args.boost_transforms
    num_xforms_mask = args.mask_transforms
    seed = args.seed
    joint_iters = args.joint_iters
    image_id = args.image_id
    beta = 1

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    
    veryBig = time.time()
    if runType == 'PrelimStats' and network == 'MNIST':
        print("Running Prelim Stats on MNIST with heatmap = ", heatmap)
        numPics = [979,1134,1031,1009,981,891,957,1027,973,1008]
        count = 0
        for hi in range(25, 100, 25): # tr_hi 
            tr_hi = hi/100
            for lo in range(0, hi+25, 25): # tr_lo 
                tr_lo = lo/100
                coarse_error = 1 - tr_hi
                reduce_error = 1 - tr_lo
                assert coarse_error <= reduce_error
                assert coarse_error >= 0
                assert reduce_error <= 1
                for num_xforms in range(0, 125, 25): # -b and -m
                    # with open("STATS/MNISTRandPrelim.txt", mode = "a") as f: 
                        # f.write(f"\nNext 50 Iterations: tr_hi = {tr_hi}, tr_lo = {tr_lo}, num_xforms = {num_xforms}") 
                    print(f"\nNext 50 Iterations: tr_hi = {tr_hi}, tr_lo = {tr_lo}, num_xforms = {num_xforms}")
                    totalTR = 0
                    totalNumPix = 0
                    totalNumQueries = 0  
                    totalTime = 0
                    for lbl_v in range(0,10): # 10 classes 
                        for j in range(5): # 5 targets (iterations) per victim 
                            lbl_t = random.choice(list(set(range(0, 9)) - set([lbl_v])))
                            numV = random.randint(0,numPics[lbl_v])
                            numT = random.randint(0,numPics[lbl_t])
                            img_v = "/home/haasf/GRAPHITE/MNIST/test/" + str(lbl_v) + "/" + str(numV) + ".png"
                            img_t = "/home/haasf/GRAPHITE/MNIST/test/" + str(lbl_t) + "/" + str(numT) + ".png"

                            image_id = "MNIST_test" + str(count)

                            timestart = time.time()
                            if(count == 792):
                                tr_score, numBits, numQ  = attack_MNIST(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarse_error, reduce_error, 
                                        beta, num_xforms_mask = num_xforms, num_xforms_boost = num_xforms, 
                                        net_size = 28, noise_size = 28, joint_iters = joint_iters, image_id = image_id)
                            else:
                                continue


                            timeend = time.time()
                            iterTime = timeend - timestart

                            # with open("STATS/MNISTRandPrelim.txt", mode = "a") as f: 
                            #     f.write("\nTotal running time: %.4f seconds" % iterTime)

                            # totalTR += tr_score
                            # totalNumPix += numBits
                            # totalNumQueries += numQ
                            # totalTime += iterTime
                            count += 1

                    avgTr = totalTR/50
                    avgNumPix = totalNumPix/50
                    avgNumQ = totalNumQueries/50
                    avgTime = totalTime/50
                    # with open("STATS/MNISTRandPrelimAvgs.txt", mode = "a") as f: 
                    #     f.write(f"\n\nStats from 50 iterations with tr_hi = {tr_hi}, tr_lo = {tr_lo}, num_xforms(b and m) = {num_xforms}")
                    #     f.write(f"\nAverage transform_robustness: {avgTr}")
                    #     f.write(f"\nAverage number of pixels: {avgNumPix}") 
                    #     f.write(f"\nAverage number of queries: {avgNumQ} ") 
                    #     f.write(f"\nAverage running time: {avgTime} seconds")


    elif runType == 'PrelimStats' and network == 'CIFAR':
        print("Running Prelim Stats w CIFAR")
        count = 0
        for hi in range(25, 100, 25): # tr_hi 
            tr_hi = hi/100
            for lo in range(0, hi+25, 25): # tr_lo 
                tr_lo = lo/100
                coarse_error = 1 - tr_hi
                reduce_error = 1 - tr_lo
                assert coarse_error <= reduce_error
                assert coarse_error >= 0
                assert reduce_error <= 1
                for num_xforms in range(0, 125, 25): # -b and -m = {0, 25, 50, 75, 100}
                    with open("STATS/CIFARPrelim.txt", mode = "a") as f: 
                        f.write(f"\nNext 50 Iterations: tr_hi = {tr_hi}, tr_lo = {tr_lo}, num_xforms = {num_xforms}") 
                    totalTR = 0
                    totalNumPix = 0
                    totalNumQueries = 0  
                    totalTime = 0
                    for lbl_v in range(0,10): # 10 classes 
                        for j in range(5): # 5 targets (iterations) per victim 
                            lbl_t = random.choice(list(set(range(0, 9)) - set([lbl_v])))
                            numV = random.randint(0,999)
                            numT = random.randint(0,999)
                            img_v = "/home/haasf/GRAPHITE/cifar/test/" + str(lbl_v) + "/" + str(numV) + ".png"
                            img_t = "/home/haasf/GRAPHITE/cifar/test/" + str(lbl_t) + "/" + str(numT) + ".png"

                            image_id = "CIFAR_test" + str(count)
                            
                            timestart = time.time()
                            tr_score, numBits, numQ  = attack_CIFAR(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarse_error, reduce_error, 
                                        beta, num_xforms_mask = num_xforms, num_xforms_boost = num_xforms, 
                                        net_size = 32, noise_size = 32, joint_iters = joint_iters, image_id = image_id)
                            timeend = time.time()
                            iterTime = timeend - timestart
                            with open("STATS/CIFARPrelim.txt", mode = "a") as f: 
                                f.write("\nTotal running time: %.4f seconds" % iterTime)

                            totalTR += tr_score
                            totalNumPix += numBits
                            totalNumQueries += numQ
                            totalTime += iterTime
                            count += 1
                    avgTr = totalTR/50
                    avgNumPix = totalNumPix/50
                    avgNumQ = totalNumQueries/50
                    avgTime = totalTime/50
                    with open("STATS/CIFARPrelimAvgs.txt", mode = "a") as f: 
                        f.write(f"\n\nStats from 50 iterations with tr_hi = {tr_hi}, tr_lo = {tr_lo}, num_xforms(b and m) = {num_xforms}")
                        f.write(f"\nAverage transform_robustness: {avgTr}")
                        f.write(f"\nAverage number of pixels: {avgNumPix}") 
                        f.write(f"\nAverage number of queries: {avgNumQ} ") 
                        f.write(f"\nAverage running time: {avgTime} seconds")

                    
    # veryEnd = time.time()
    # with open("STATS/MNISTPrelimAvgs.txt", mode = "a") as f: 
    #     f.write(f"\n Total time for all {count} iterations: {veryEnd - veryBig}")

    
    
    

    timestart = time.time()

    
    if network == 'GTSRB':
        attack_GTSRB(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, args.coarse_error, args.reduce_error, beta, num_xforms_mask, num_xforms_boost, net_size = 32, noise_size = 32, joint_iters = joint_iters)
    elif network == 'CIFAR':
        attack_CIFAR(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, args.coarse_error, args.reduce_error, beta, num_xforms_mask, num_xforms_boost, net_size = 32, noise_size = 32, joint_iters = joint_iters, image_id = image_id)
    elif network == 'MNIST':
        attack_MNIST(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, args.coarse_error, args.reduce_error, beta, num_xforms_mask, num_xforms_boost, net_size = 28, noise_size = 28, joint_iters = joint_iters, image_id = image_id)
    timeend = time.time()

    with open("STATS/MNIST.txt", mode = "a") as f: 
        f.write("\nTotal running time: %.4f seconds" % (timeend - timestart))

    
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))
