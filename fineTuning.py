import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Resnet50FPN, CountRegressor, weights_normal_init
from utils import MAPS, Scales, TransformTrain, extract_features
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm
import argparse
import random

parser = argparse.ArgumentParser(description="Fine-tune a pre-trained counting model")
parser.add_argument("-dp", "--data_path", type=str, required=True, help="Path to the FSC147 dataset")
parser.add_argument("-o", "--output_dir", type=str, default="./logsFineTune", help="Path to output logs")
parser.add_argument("-ts", "--test_split", type=str, default='val', choices=["train", "test", "val"], help="Data split for evaluation")
parser.add_argument("-ep", "--epochs", type=int, default=500, help="Number of fine-tuning epochs")
parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU id")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-6, help="Fine-tuning learning rate")
parser.add_argument("-m", "--model_path", type=str, default="./data-final/FamNet_Save1.pth", help="Path to the pre-trained model")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

criterion = nn.MSELoss().cuda()
resnet50_conv = Resnet50FPN().cuda()
regressor = CountRegressor(6, pool='mean').cuda()

checkpoint = torch.load(args.model_path)
print(checkpoint.keys())
regressor.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.Adam(regressor.parameters(), lr=args.learning_rate)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.data_path, 'annotations.json')) as f:
    annotations = json.load(f)

with open(os.path.join(args.data_path, 'Train_Test_Val.json')) as f:
    data_split = json.load(f)

def fine_tune():
    print("Fine-tuning on FSC147 train set")
    im_ids = data_split['train']
    random.shuffle(im_ids)
    best_mae = float('inf')

    for epoch in range(args.epochs):
        train_mae, train_rmse, train_loss = 0, 0, 0
        pbar = tqdm(im_ids)
        for im_id in pbar:
            anno = annotations[im_id]
            bboxes = anno['box_examples_coordinates']
            dots = np.array(anno['points'])
            rects = [[y1, x1, y2, x2] for x1, y1, x2, y2 in [bbox[0] + bbox[2] for bbox in bboxes]]
            
            image = Image.open(os.path.join(args.data_path, 'indt-objects-V4', im_id))
            density = np.load(os.path.join(args.data_path, 'density_map_adaptive_V1', im_id.replace(".jpg", ".npy"))).astype('float32')
            
            sample = {'image': image, 'lines_boxes': rects, 'gt_density': density}
            sample = TransformTrain(sample)
            image, boxes, gt_density = sample['image'].cuda(), sample['boxes'].cuda(), sample['gt_density'].cuda()
            
            with torch.no_grad():
                features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
            features.requires_grad = True
            optimizer.zero_grad()
            output = regressor(features)
            
            if output.shape[2:] != gt_density.shape[2:]:
                gt_density = F.interpolate(gt_density, size=output.shape[2:], mode='bilinear')
            
            loss = criterion(output, gt_density)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred_cnt, gt_cnt = torch.sum(output).item(), torch.sum(gt_density).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            train_mae += cnt_err
            train_rmse += cnt_err ** 2
            
            pbar.set_description(f"Epoch {epoch+1}/{args.epochs}: actual-predicted: {gt_cnt:.1f}, {pred_cnt:.1f}, error: {cnt_err:.1f}. MAE: {train_mae/len(im_ids):.2f}, RMSE: {(train_rmse/len(im_ids))**0.5:.2f}")

        if train_mae / len(im_ids) < best_mae:
            best_mae = train_mae / len(im_ids)
            torch.save({'model_state_dict': regressor.state_dict()}, os.path.join(args.output_dir, 'best_finetuned_model.pth'))
        
if __name__ == "__main__":
    fine_tune()
