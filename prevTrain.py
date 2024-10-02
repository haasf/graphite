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
from torch.utils.data import DataLoader
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
    # tr_hi = 0.50
    # tr_lo = 0.25
    num_training_xforms = 10
    num_testing_xforms = 100
    # heatmap = 'Random'

    # Largest index per class in MNIST/train (ex 5922.png for class 0)
    train_counts = [5922,6741,5957,6130,5841,5420,5917,6264,5850,5948]
    test_counts = [979,1134,1031,1009,981,891,957,1027,973,1008]

    # testDS = datasets.MNIST(root='.', train=False, download=True, transform=transforms.ToTensor())
    # testDL = DataLoader(testDS, shuffle= True, batch_size= 50) # load 50 random test imgs


    

    ds = datasets.MNIST(root='.', train=True, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds,shuffle=True)

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

    
    for img_v, lbl_v_ten in tqdm(dl):
        print(f"countTrained = {countTrained}\n")
        lbl_v = lbl_v_ten.cpu().numpy()[0]

        print("lbl_v: ", lbl_v, " model.forward(img_v): ", (model.forward(img_v.cuda()).cpu().detach().numpy())[0])
        save_path = "modelTraining/MNIST/tempVictimImg.png"
        save_image(img_v,save_path)

        lbl_t = random.choice(list(set(range(0, 9)) - set([lbl_v])))
        numT = random.randint(0,train_counts[lbl_t])

        img_t = "/home/haasf/GRAPHITE/MNIST/train/" + str(lbl_t) + "/" + str(numT) + ".png"
        
        
        image_id = "MNIST_Training_Number_" + str(countTrained)

        # print("coarseerror: ", coarseerror, "reduceerror: ", reduceerror, "beta: ", beta, "num_xforms: ", num_training_xforms)
        # timeNow = time.time()
        attackedImg = attack_network(model = model, img_v = save_path, img_t = img_t, mask = mask, lbl_v = lbl_v, lbl_t = lbl_t, 
                pt_file = pt_file, scorefile = scorefile, heatmap = 'Random', coarseerror = coarseerror, reduceerror = reduceerror, beta = beta, 
                num_xforms_mask = num_training_xforms, num_xforms_boost= num_training_xforms, net_size = 28, noise_size = 28, model_type = 'MNIST', joint_iters= joint_iters, image_id = image_id, return_type = 'Image')
        # timeLater = time.time()
        # iterTime = timeLater - timeNow
    
        # with open("modelTraining/MNIST/training.txt", mode = "a") as f: 
        #     f.write(f"\nIteration # {countTrained}, running time: %.4f seconds" % iterTime)

        # test_lbl_t = lbl_t 
        # test_lbl_v = lbl_v 
        # test_img_v = save_path
        # test_img_t = img_t
        # tr_score, numBits, numQ = attack_network(model = model, img_v = test_img_v,img_t = test_img_t, mask = mask, lbl_v = test_lbl_v, lbl_t = test_lbl_t, 
        #             pt_file = pt_file, scorefile = scorefile, heatmap = 'Target', coarseerror = coarseerror, reduceerror = reduceerror, beta = beta, 
        #             num_xforms_mask = 100, num_xforms_boost= 100, net_size = 28, noise_size = 28, model_type = 'MNIST', joint_iters= joint_iters, image_id = image_id)
        
        # with open("modelTraining/MNIST/TestResults.txt", mode = "a") as f: 
        #     f.write("\n\nStats before training: \n")
        #     f.write(f"tr_score: {tr_score}, numBits: {numBits}, numQueries: {numQ}\n")

        optimizer.zero_grad()
        attackedImg = attackedImg[np.newaxis,:,:,:].cuda()
        pred = model.forward(attackedImg)
        # pred = model.getProbs(attackedImg).requires_grad_()
        loss = criterion(pred, lbl_v_ten.cuda())
        loss.backward()
        optimizer.step()    
        
        

        # save model every 100 iterations, check on a few of the images from the past 100 
        # save img_v and lbl_v_ten every 10 iterations 
        temp = countTrained % 100 
        
        # if temp % 10 == 0:
            # path = "modelTraining/MNIST/attacked" + str(int(temp / 10)) + ".png"
            # save_image(attackedImg, path)   
               
        if temp == 10:
            tenLbl_v[1] = lbl_v 
            attacked1 = attackedImg
        elif temp == 20:
            tenLbl_v[2] = lbl_v 
            attacked2 = attackedImg
        elif temp == 30:
            tenLbl_v[3] = lbl_v 
            attacked3 = attackedImg
        elif temp == 40:
            tenLbl_v[4] = lbl_v 
            attacked4 = attackedImg
        elif temp == 50:
            tenLbl_v[5] = lbl_v 
            attacked5 = attackedImg
        elif temp == 60:
            tenLbl_v[6] = lbl_v 
            attacked6 = attackedImg
        elif temp == 70:
            tenLbl_v[7] = lbl_v 
            attacked7 = attackedImg
        elif temp == 80:
            tenLbl_v[8] = lbl_v 
            attacked8 = attackedImg
        elif temp == 90:
            tenLbl_v[9] = lbl_v 
            attacked9 = attackedImg
        elif temp == 0:
            if(countTrained > 0): # not the very start
                attacked = [attacked0,attacked1,attacked2,attacked3,attacked4,attacked5,attacked6,attacked7,attacked8,attacked9]
                with open("modelTraining/MNIST/TestResults.txt", mode="a") as f:
                    f.write(f"\nTesting model on 10 of last 100 training adversarial examples\n")
                    for i in range(10):
                        # pred = model.predict(attacked[i])
                        probs = (model.forward(attacked[i].cuda()).cpu().detach().numpy())[0]
                        f.write(f"Prob for lbl_v = {tenLbl_v[i]}, {probs[tenLbl_v[i]]}\n")
                        f.write(f"all probs: {probs}\n\n")
                        # if(pred == tenLbl_v[i]):
                            # f.write(f"SUCCESS: pred = lbl_v = {pred}\n")
                        # else: 
                        #     f.write(f"FAIL: pred = {pred} not equal to lbl_v = {tenLbl_v[i]}\n")

                    

                quit()
                        
            tenLbl_v[0] = lbl_v
            attacked0 = attackedImg
    

        ### check if new example is no longer misclassified 
        # model.predict(attackedImg) should = lbl_v
        # if(model.predict(attackedImg) != lbl_v):
        #     print("model.predict: ", model.predict(attackedImg), "not equal to ", lbl_v)
        #     quit()
        # else:
        #     print("SUCCESS, model.predict: ", model.predict(attackedImg), "= lbl_v: ", lbl_v)

        # # test on same image: 
        # image_id = "MNIST_8/5SelfTest_" + str(countTrained)
        # attackedImg = attack_network(model = model, img_v = save_path, img_t = img_t, mask = mask, lbl_v = lbl_v, lbl_t = lbl_t, 
        #         pt_file = pt_file, scorefile = scorefile, heatmap = 'Random', coarseerror = coarseerror, reduceerror = reduceerror, beta = beta, 
        #         num_xforms_mask = num_training_xforms, num_xforms_boost= num_training_xforms, net_size = 28, noise_size = 28, model_type = 'MNIST', joint_iters= joint_iters, image_id = image_id, return_type = 'Image')
        # save_image(attackedImg, "modelTraining/MNIST/attackedAfterTrained.png")

        # tr_score, numBits, numQ = attack_network(model = model, img_v = test_img_v,img_t = test_img_t, mask = mask, lbl_v = test_lbl_v, lbl_t = test_lbl_t, 
        #             pt_file = pt_file, scorefile = scorefile, heatmap = 'Target', coarseerror = coarseerror, reduceerror = reduceerror, beta = beta, 
        #             num_xforms_mask = 100, num_xforms_boost= 100, net_size = 28, noise_size = 28, model_type = 'MNIST', joint_iters= joint_iters, image_id = image_id)

    




        countTrained += 1
        ##### Test 50 unseen images from test set
        # if(countTrained % 50 == 0):
        #     with open("modelTraining/MNIST/TestResults.txt", mode = "a") as f: 
        #         f.write(f"\nTesting: countTrained = {countTrained}, countTests = {countTests}")
        #     totalTR = 0
        #     totalNumPix = 0
        #     totalNumQueries = 0  
        #     totalTime = 0
        #     for i in range(50):
        #         test_lbl_t = testTgts[i]
        #         test_lbl_v = testVics[i]
        #         test_img_v = "MNIST/test50Victims/" + str(i) + ".png"
        #         test_img_t = "MNIST/test50Targets/" + str(i) + ".png"
        #         image_id = "MNIST_Test_Number_" + str(countTests)

        #         timestart = time.time()
        #         tr_score, numBits, numQ = attack_network(model = model, img_v = test_img_v,img_t = test_img_t, mask = mask, lbl_v = test_lbl_v, lbl_t = test_lbl_t, 
        #             pt_file = pt_file, scorefile = scorefile, heatmap = 'Target', coarseerror = coarseerror, reduceerror = reduceerror, beta = beta, 
        #             num_xforms_mask = 100, num_xforms_boost= 100, net_size = 28, noise_size = 28, model_type = 'MNIST', joint_iters= joint_iters, image_id = image_id)
        #         timeend = time.time()
        #         iterTime = timeend - timestart
        #         totalTR += tr_score
        #         totalNumPix += numBits
        #         totalNumQueries += numQ
        #         totalTime += iterTime
        #         countTests += 1
                    
        #     avgTr = totalTR/50
        #     avgNumPix = totalNumPix/50
        #     avgNumQ = totalNumQueries/50
        #     avgTime = totalTime/50

        #     with open("modelTraining/MNIST/TestResults.txt", mode="a") as f:
        #         f.write(f"\n\nTest Statistics after training for {countTrained} iterations, test # {countTests}")
        #         f.write(f"\nAverage transform_robustness: {avgTr}")
        #         f.write(f"\nAverage number of pixels: {avgNumPix}") 
        #         f.write(f"\nAverage number of queries: {avgNumQ} ") 
        #         f.write(f"\nAverage running time: {avgTime} seconds")



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
