import time
import random 
import numpy as np
import cv2
import sys
from utils import run_predictions, grayDim
import torch 
from MNIST.mnist_net import MnistNet
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import parsearguments
from tqdm import tqdm

from main import attack_network


def train_MNIST(mask, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta):
    net = MnistNet()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
    
    net.eval()

    model = net.module if torch.cuda.is_available() else net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('MNIST/epoch-39-MNIST.ckpt', map_location=device)
    model.load_state_dict(checkpoint)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

    # SETTINGS
    num_training_xforms = 10
    train_counts = [5922,6741,5957,6130,5841,5420,5917,6264,5850,5948]
    test_counts = [979,1134,1031,1009,981,891,957,1027,973,1008]
    indices = random.sample(range(0,60000),100)
    ds = datasets.MNIST(root='.', train=True, download=True, transform=transforms.ToTensor())
    # subset = Subset(ds,indices)
    dl = DataLoader(ds,shuffle=False)

    countTrained = 0
    countTests = 0
    # Classes for MNIST/test50Targets
    testTgts = np.array([0, 5, 0, 7, 0, 9, 8, 7, 1, 5, 
                         3, 9, 1, 3, 5, 3, 2, 2, 6, 3, 
                         0, 4, 3, 1, 0, 6, 1, 6, 6, 8, 
                         2, 4, 5, 8, 2, 9, 6, 4, 4, 8, 
                         4, 5, 1, 7, 9, 7, 9, 2, 8, 7])
    # Classes for MNIST/test50Victims
    testVics = np.array([7, 1, 1, 3, 2, 8, 5, 8, 5, 1, 
                         8, 3, 9, 2, 9, 4, 6, 9, 9, 0, 
                         6, 0, 0, 5, 3, 7, 2, 4, 1, 6, 
                         4, 2, 0, 1, 6, 8, 5, 7, 5, 0, 
                         2, 4, 9, 8, 7, 4, 6, 3, 7, 3])
    
    testIndices = random.sample(range(0,10000),50)
    testDs = datasets.MNIST(root='.', train=False, download=True, transform=transforms.ToTensor())
    testsubset = Subset(testDs,testIndices)
    testdl = DataLoader(testsubset,shuffle=False)
    state = random.getstate()
    numSkipped = 0
    for img_v, lbl_v_ten in tqdm(dl):
        if countTrained % 50 == 0:
            print("countTrained % 50 == 0: Testing on 50 test examples")
            random.setstate(state)
            testSkips = 0
            numTested = 0
            totalTR = 0
            totalBits = 0
            totalQueries = 0
            totalTime = 0
            for test_img_v, test_lbl_v_ten in tqdm(testdl):
                test_lbl_v = test_lbl_v_ten.cpu().numpy()[0]
                test_lbl_t = random.choice(list(set(range(0, 9)) - set([test_lbl_v])))
                test_numT = random.randint(0,test_counts[test_lbl_t])
                test_img_t = "/home/haasf/GRAPHITE/MNIST/test/" + str(test_lbl_t) + "/" + str(test_numT) + ".png"

                image_id = "MNIST_Testing_Number_" + str(countTrained)
                test_save_path ="modelTraining/MNIST/testTempVicImg.png"
                save_image(test_img_v,test_save_path)

               

                timeStart = time.time()
                tr_score, nbits, numQueries = attack_network(model = model, img_v = test_save_path, img_t = test_img_t, mask = mask, lbl_v = test_lbl_v, lbl_t = test_lbl_t, 
                pt_file = pt_file, scorefile = scorefile, heatmap = 'Random', coarseerror = coarseerror, reduceerror = reduceerror, beta = beta, 
                num_xforms_mask = num_training_xforms, num_xforms_boost= num_training_xforms, net_size = 28, noise_size = 28, model_type = 'MNIST', joint_iters= joint_iters, image_id = image_id)
                timeEnd = time.time()
        
                timeIter = timeEnd - timeStart

                # if(nbits < 78.4):
                totalTR += tr_score
                totalBits += nbits
                totalQueries += numQueries
                totalTime += timeIter
                numTested += 1
                # else: 
                #     testSkips += 1

            # assert((numTested + testSkips) == 50)
            
            with open ("modelTraining/MNIST/TestResults.txt", mode="a") as f:
                f.write(f"\nTest Statistics after training for {countTrained} iterations")
                f.write(f"\nAverage TR: {(totalTR/numTested):.3f}")
                f.write(f"\nAverage nBits: {(totalBits/numTested):.2f}")
                f.write(f"\nAverage numQ: {(totalQueries/numTested):.1f}")
                f.write(f"\nAverage time: {(totalTime/numTested):.3f} seconds")
                # f.write(f"\n\Testing images skipped (nbits > 78.4): {testSkips}")
                f.write(f"\nTraining images skipped (nbits > 78.4): {numSkipped}")

            numSkipped = 0
            # end if 


        print(f"countTrained = {countTrained}\n")
        lbl_v = lbl_v_ten.cpu().numpy()[0]
        save_path = "modelTraining/MNIST/tempVictimImg.png"
        save_image(img_v,save_path)

        lbl_t = random.choice(list(set(range(0, 9)) - set([lbl_v])))
        numT = random.randint(0,train_counts[lbl_t])
        img_t = "/home/haasf/GRAPHITE/MNIST/train/" + str(lbl_t) + "/" + str(numT) + ".png"
        image_id = "MNIST_Training_Number_" + str(countTrained)

    
        attackedImg, nBits = attack_network(model = model, img_v = save_path, img_t = img_t, mask = mask, lbl_v = lbl_v, lbl_t = lbl_t, 
                pt_file = pt_file, scorefile = scorefile, heatmap = 'Random', coarseerror = coarseerror, reduceerror = reduceerror, beta = beta, 
                num_xforms_mask = num_training_xforms, num_xforms_boost= num_training_xforms, net_size = 28, noise_size = 28, model_type = 'MNIST', joint_iters= joint_iters, image_id = image_id, return_type = 'Image')

        
        if(nBits < 78.4):
            optimizer.zero_grad()
            attackedImg = attackedImg[np.newaxis,:,:,:].cuda()
            pred = model(attackedImg)
            loss = criterion(pred, lbl_v_ten.cuda())
            loss.backward()
            optimizer.step() 
        else:
            numSkipped += 1

        
        
          
            

        countTrained += 1
        



if __name__ == '__main__':



    network = 'MNIST'

    args = parsearguments.getarguments()
    network = args.network
    # img_v = args.img_v
    # img_t = args.img_t
    mask = args.mask
    # lbl_v = args.lbl_v
    # lbl_t = args.lbl_t
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


    if(network == 'MNIST'):
        train_MNIST(mask, pt_file, scorefile, heatmap, args.coarse_error, args.reduce_error, beta)
