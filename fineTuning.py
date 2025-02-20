import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Resnet50FPN, CountRegressor
from utils import MAPS, Scales, TransformTrain, extract_features
from PIL import Image
import json
import numpy as np
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Fine-Tuning Few Shot Counting Model")
parser.add_argument("-dp", "--data_path", type=str, required=True, help="Path to the FSC147 dataset")
parser.add_argument("-mp", "--model_path", type=str, required=True, help="Path to the pre-trained .pth model")
parser.add_argument("-o", "--output_dir", type=str, default="./logsFineTuned", help="Output directory")
parser.add_argument("-ep", "--epochs", type=int, default=500, help="Number of fine-tuning epochs")
parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU ID")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-6, help="Fine-tuning learning rate")
args = parser.parse_args()

# Setup GPU
torch.cuda.set_device(args.gpu)

# Load pre-trained model
resnet50_conv = Resnet50FPN().cuda()
resnet50_conv.eval()

regressor = CountRegressor(6, pool='mean').cuda()
regressor.load_state_dict(torch.load(args.model_path))
regressor.train()

optimizer = optim.Adam(regressor.parameters(), lr=args.learning_rate)
criterion = nn.MSELoss().cuda()

# Load dataset
with open(os.path.join(args.data_path, 'annotations.json')) as f:
    annotations = json.load(f)
with open(os.path.join(args.data_path, 'Train_Test_Val.json')) as f:
    data_split = json.load(f)

def fine_tune():
    print("Fine-tuning on FSC147 train set")
    im_ids = data_split['train']
    best_mae = float('inf')
    best_rmse = float('inf')
    
    for epoch in range(args.epochs):
        train_loss, train_mae, train_rmse = 0, 0, 0
        random.shuffle(im_ids)
        pbar = tqdm(im_ids)
        
        for im_id in pbar:
            anno = annotations[im_id]
            bboxes = anno['box_examples_coordinates']
            rects = [[bbox[0][1], bbox[0][0], bbox[2][1], bbox[2][0]] for bbox in bboxes]
            
            image = Image.open(f'{args.data_path}/indt-objects-V4/{im_id}').convert('RGB')
            density = np.load(f'{args.data_path}/density_map_adaptive_V1/{im_id.split(".jpg")[0]}.npy').astype('float32')
            
            sample = TransformTrain({'image': image, 'lines_boxes': rects, 'gt_density': density})
            image, boxes, gt_density = sample['image'].cuda(), sample['boxes'].cuda(), sample['gt_density'].cuda()
            
            with torch.no_grad():
                features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
            
            features.requires_grad = True
            optimizer.zero_grad()
            output = regressor(features)
            
            if output.shape[2:] != gt_density.shape[2:]:
                gt_density = F.interpolate(gt_density, size=output.shape[2:], mode='bilinear', align_corners=False)
            
            loss = criterion(output, gt_density)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += abs(torch.sum(output).item() - torch.sum(gt_density).item())
            train_rmse += (train_mae ** 2)
            
        train_mae /= len(im_ids)
        train_rmse = (train_rmse / len(im_ids)) ** 0.5
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={train_loss:.4f}, MAE={train_mae:.2f}, RMSE={train_rmse:.2f}")
        
        if train_mae < best_mae:
            best_mae = train_mae
            best_rmse = train_rmse
            torch.save(regressor.state_dict(), os.path.join(args.output_dir, 'fine_tuned_model.pth'))
            print("Saved best fine-tuned model.")

if __name__ == "__main__":
    fine_tune()
