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

    indices = random.sample(range(0,60000),100)
    ds = datasets.MNIST(root='.', train=True, download=True, transform=transforms.ToTensor())
    subset = Subset(ds,indices)
    dl = DataLoader(subset,shuffle=False)

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
    
    tenLbl_v = [10,10,10,10,10,10,10,10,10,10]
    tenLbl_t = [10,10,10,10,10,10,10,10,10,10]
    state = random.getstate()
    smax = torch.nn.Softmax(dim=0)
    for i in range(5):
        random.setstate(state)

        for img_v, lbl_v_ten in tqdm(dl):
            print(f"i = {i}, countTrained = {countTrained}\n")
            lbl_v = lbl_v_ten.cpu().numpy()[0]
            save_path = "modelTraining/MNIST/tempVictimImg.png"
            save_image(img_v,save_path)

            lbl_t = random.choice(list(set(range(0, 9)) - set([lbl_v])))
            numT = random.randint(0,train_counts[lbl_t])
            img_t = "/home/haasf/GRAPHITE/MNIST/train/" + str(lbl_t) + "/" + str(numT) + ".png"
            image_id = "MNIST_Training_Number_" + str(countTrained)
            attackedImg, nbits, tr_score = attack_network(model = model, img_v = save_path, img_t = img_t, mask = mask, lbl_v = lbl_v, lbl_t = lbl_t, 
                    pt_file = pt_file, scorefile = scorefile, heatmap = 'Random', coarseerror = coarseerror, reduceerror = reduceerror, beta = beta, 
                    num_xforms_mask = num_training_xforms, num_xforms_boost= num_training_xforms, net_size = 28, noise_size = 28, model_type = 'MNIST', joint_iters= joint_iters, image_id = image_id, return_type = 'Image')


            if nbits == 0:
                pred = model.predict(img_v.cuda())
                with open("modelTraining/MNIST/TestResults.txt", mode="a") as f:
                    f.write(f"attacked nbits = 0, checking img_v prediction\n")
                    if(pred == lbl_v):
                        f.write(f"CORRECTLY CLASSIFIED BUT NBITS = 0\n")
                    else:  
                        f.write(f"Incorrectly classified, pred = {pred} != lbl_v = {lbl_v}\n")


            optimizer.zero_grad()
            attackedImg = attackedImg[np.newaxis,:,:,:].cuda()
            pred = model(attackedImg)
            loss = criterion(pred, lbl_v_ten.cuda())
            loss.backward()
            optimizer.step()    
            
            optimizer.zero_grad()
            pred = model(img_v.cuda())
            loss = criterion(pred, lbl_v_ten.cuda())
            loss.backward()
            optimizer.step()   
            temp = countTrained % 100 

            if (temp % 10) == 0:
                imgNum = int(temp/10)
                if (i == 0):
                    tenLbl_v[imgNum] = lbl_v
                    tenLbl_t[imgNum] = lbl_t
                else:
                    assert(tenLbl_v[imgNum] == lbl_v)
                    assert(tenLbl_t[imgNum] == lbl_t)

                save_path = "modelTraining/MNIST/Task2/" + str(imgNum) + "/" + str(i) + ".png"
                save_image(attackedImg,save_path)
    
                with open("modelTraining/MNIST/TestResults.txt", mode="a") as f:
                    f.write(f"i = {i}, attacked{imgNum}: nbits = {nbits}, TR = {tr_score}\n")
            # base path = modelTraining/MNIST/Task2/ + {attacked number}/{img number}.png
            basePath ="modelTraining/MNIST/Task2/"
            if temp == 10:
                attacked1 = attackedImg
            elif temp == 20:
                attacked2 = attackedImg
            elif temp == 30:
                attacked3 = attackedImg
            elif temp == 40:
                attacked4 = attackedImg
            elif temp == 50:
                attacked5 = attackedImg
            elif temp == 60:
                attacked6 = attackedImg
            elif temp == 70:
                attacked7 = attackedImg
            elif temp == 80:
                attacked8 = attackedImg
            elif temp == 90:
                attacked9 = attackedImg
            elif temp == 0:
                attacked0 = attackedImg
            
            if (temp == 0 and countTrained > 0): # not the very start

                    

                attacked = [attacked0,attacked1,attacked2,attacked3,attacked4,attacked5,attacked6,attacked7,attacked8,attacked9]
                with open("modelTraining/MNIST/TestResults.txt", mode="a") as f:
                    f.write(f"i = {i}, Testing model on 10 of last 100 training adversarial examples\n")
                    for j in range(10):
                        pred = model.predict(attacked[j])
                        probs = smax(((model.forward(attacked[j].cuda()))[0])).data.cpu().detach().numpy()
                        # f.write(f"Prob for lbl_v = {tenLbl_v[j]:.3f} = {probs[tenLbl_v[j]]:.3f}, highest prob = {np.max(probs):.3f} for index {np.argmax(probs)}\n")
                        if(pred == tenLbl_v[j]):
                            f.write(f"attacked{j}: prob lbl_v = {tenLbl_v[j]} = {probs[tenLbl_v[j]]}, prob lbl_t = {tenLbl_t[j]} = {probs[tenLbl_t[j]]}.\n SUCCESS: Prob for lbl_v = {tenLbl_v[j]} = {probs[tenLbl_v[j]]:.3f} is the highest prob\n")
                        else: 
                            f.write(f"attacked{j}: prob lbl_v = {tenLbl_v[j]} = {probs[tenLbl_v[j]]}, prob lbl_t = {tenLbl_t[j]} = {probs[tenLbl_t[j]]}.\n FAIL: Prob for lbl_v = {tenLbl_v[j]} = {probs[tenLbl_v[j]]:.3f}, highest prob = {np.max(probs):.3f} for class {np.argmax(probs)}\n")
                        f.write(f"all probs: {probs}\n\n")

                    f.write(f"i = {i}, Testing on 100 unattacked images \n")
                    num = 0
                    for img_v, lbl_v_ten in tqdm(dl):
                        lbl_v = lbl_v_ten.cpu().numpy()[0]
                        pred = model.predict(img_v)
                        if(pred == lbl_v):
                            f.write(f"num: {num}, success: pred: {pred} = lbl_v: {lbl_v}\n")
                        else:
                            f.write(f"num: {num}, fail: pred: {pred}, lbl_v = {lbl_v}\n\n")

                        num += 1
                    

                    

                            
                

            countTrained += 1
        
        with open ("modelTraining/MNIST/training.txt", mode="a") as g:
            g.write(f"i = {i}, tenLbl_v = {tenLbl_v}, tenLbl_t = {tenLbl_t}\n")


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
