import torch
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

if __name__ == '__main__':
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    datasets.MNIST.resources = [
       ('/'.join([new_mirror, url.split('/')[-1]]), md5)
       for url, md5 in datasets.MNIST.resources
    ]
    counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    
    testDS =  ds = datasets.MNIST(root='.', train=False, download=True, transform=transforms.ToTensor())
    testDL = DataLoader(testDS, shuffle= True, batch_size= 50) # load 50 random test imgs

   

    # ds = datasets.MNIST(root='.', train=False, download=True, transform=transforms.ToTensor())
    # dl = DataLoader(ds)
    # if not os.path.exists('test/'):
    #     os.makedirs('test')
    #     for i in range(10):
    #         os.makedirs('test/{}'.format(i))
    # for x, y, in tqdm(dl):
    #     y = y.cpu().numpy()[0]
    #     save_path = 'test/{}'.format(y)
    #     imgs_list = os.listdir(save_path)
    #     imgs_list.sort()
    #     last_img_idx = counts[y]
    #     counts[y] += 1
    #     save_path = 'test/{}/{}.png'.format(y, last_img_idx)
    #     save_image(x, save_path)
    

    # # Get Training Data
    # ds = datasets.MNIST(root='.', train=True, download=True, transform=transforms.ToTensor())
    # dl = DataLoader(ds)
    # if not os.path.exists('train/'):
    #     os.makedirs('train')
    #     for i in range(10):
    #         os.makedirs('train/{}'.format(i))
    # for x, y, in tqdm(dl):
    #     y = y.cpu().numpy()[0]
    #     save_path = 'train/{}'.format(y)
    #     imgs_list = os.listdir(save_path)
    #     imgs_list.sort()
    #     last_img_idx = counts[y]
    #     counts[y] += 1
    #     save_path = 'train/{}/{}.png'.format(y, last_img_idx)
    #     save_image(x, save_path)
