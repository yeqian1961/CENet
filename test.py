import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from net import CENet
from utils.data_cod import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--test_path',type=str,default='./Dataset/TestDataset/',help='test dataset path')
opt = parser.parse_args()
dataset_path = opt.test_path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = CENet()
model.load_state_dict(torch.load('./model_pth/CENet_best.pth'))
model.cuda()
model.eval()

test_datasets = ['CAMO','CHAMELEON','COD10K', 'NC4K']
for dataset in test_datasets:
    save_path = './test_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/Imgs/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        _, _, _, res4 = model(image)
        res = F.interpolate(res4, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res*255)
