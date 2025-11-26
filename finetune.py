"""
Finetuning Code for Few-Shot Object Counting Model on Domain-Specific Data
Based on: Learning To Count Everything, CVPR 2021
Authors: Viresh Ranjan, Udbhav, Thu Nguyen, Minh Hoai

Modified for domain-specific finetuning
Date: 2024
"""
import torch.nn as nn
from model import Resnet50FPN, CountRegressor, weights_normal_init
from utils import MAPS, Scales, Transform, TransformTrain, extract_features, visualize_output_and_save
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists, join
import random
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime


def setup_finetune_parser():
    """Setup argument parser for finetuning"""
    parser = argparse.ArgumentParser(description="Finetune Few-Shot Counting Model on Domain-Specific Data")
    
    # Data paths
    parser.add_argument("-dp", "--data_path", type=str, default='./data-final/', 
                        help="Path to domain-specific dataset")
    parser.add_argument("-anno", "--annotation_file", type=str, default='./data-final/annotations.json',
                        help="Path to annotations JSON file")
    parser.add_argument("-split", "--split_file", type=str, default='./data-final/Train_Test_Val.json',
                        help="Path to train/test/val split JSON file")
    parser.add_argument("-img_dir", "--image_dir", type=str, default='./data-final/indt-objects-V4',
                        help="Directory containing images")
    parser.add_argument("-gt_dir", "--gt_dir", type=str, default='./data-final/density_map_adaptive_V1',
                        help="Directory containing ground truth density maps")
    
    # Model checkpoints
    parser.add_argument("-ckpt", "--checkpoint", type=str, default=None,
                        help="Path to pretrained CountRegressor checkpoint to load")
    parser.add_argument("-o", "--output_dir", type=str, default="./logsSave",
                        help="Directory to save finetuned models and logs")
    
    # Training parameters
    parser.add_argument("-ep", "--epochs", type=int, default=100,
                        help="Number of finetuning epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5,
                        help="Learning rate for finetuning")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0,
                        help="Weight decay for optimizer")
    
    # Finetuning strategy
    parser.add_argument("-freeze_backbone", "--freeze_backbone", type=bool, default=False,
                        help="Freeze ResNet50 backbone during finetuning")
    parser.add_argument("-freeze_conv_layers", "--freeze_conv_layers", type=int, default=0,
                        help="Number of conv layers to freeze in FPN (0=freeze none, 1=freeze conv1, etc)")
    
    # Evaluation
    parser.add_argument("-ts", "--test_split", type=str, default='val', 
                        choices=["train", "test", "val"],
                        help="Data split to evaluate on during training")
    parser.add_argument("-eval_freq", "--eval_frequency", type=int, default=5,
                        help="Evaluate every N epochs")
    
    # Hardware
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="GPU id to use")
    
    # Miscellaneous
    parser.add_argument("-seed", "--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("-save_freq", "--save_frequency", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    return parser.parse_args()


class FinetunedCountingModel:
    """Wrapper class for finetuning the counting model"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        
        # Setup paths
        if not exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not exists(join(args.output_dir, "stats")):
            os.makedirs(join(args.output_dir, "stats"))
        if not exists(join(args.output_dir, "checkpoints")):
            os.makedirs(join(args.output_dir, "checkpoints"))
        
        # Set random seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Load models
        self._load_models()
        
        # Load data
        self._load_data()
        
        # Setup training
        self._setup_training()
        
        self.best_mae = float('inf')
        self.best_rmse = float('inf')
        self.training_stats = []
        
        print(f"✓ Model initialized on device: {self.device}")
    
    def _load_models(self):
        """Load ResNet50-FPN and CountRegressor"""
        print("Loading models...")
        
        # Load feature extractor (ResNet50-FPN)
        self.resnet50_conv = Resnet50FPN()
        self.resnet50_conv = self.resnet50_conv.to(self.device)
        self.resnet50_conv.eval()
        
        # Load count regressor
        self.regressor = CountRegressor(6, pool='mean')
        
        # Load pretrained checkpoint if provided
        if self.args.checkpoint and exists(self.args.checkpoint):
            print(f"Loading pretrained regressor from: {self.args.checkpoint}")
            checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
            self.regressor.load_state_dict(checkpoint)
        else:
            print("Initializing regressor with normal distribution")
            weights_normal_init(self.regressor, dev=0.001)
        
        self.regressor = self.regressor.to(self.device)
        self.regressor.train()
        
        # Optionally freeze backbone
        if self.args.freeze_backbone:
            print("Freezing ResNet50 backbone")
            for param in self.resnet50_conv.parameters():
                param.requires_grad = False
    
    def _load_data(self):
        """Load annotation and split information"""
        print("Loading data annotations...")
        
        with open(self.args.annotation_file) as f:
            self.annotations = json.load(f)
        
        with open(self.args.split_file) as f:
            self.data_split = json.load(f)
        
        print(f"✓ Loaded {len(self.annotations)} annotations")
        print(f"✓ Train images: {len(self.data_split.get('train', []))}")
        print(f"✓ Val images: {len(self.data_split.get('val', []))}")
        print(f"✓ Test images: {len(self.data_split.get('test', []))}")
    
    def _setup_training(self):
        """Setup loss function and optimizer"""
        self.criterion = nn.MSELoss()
        
        # Setup optimizer with optional weight decay
        self.optimizer = optim.Adam(
            self.regressor.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Optional: Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )
    
    def _load_image_and_density(self, im_id):
        """Load image and ground truth density map"""
        image_path = join(self.args.image_dir, im_id)
        image = Image.open(image_path)
        image.load()
        
        density_filename = im_id.split(".")[0] + ".npy"
        density_path = join(self.args.gt_dir, density_filename)
        
        if not exists(density_path):
            raise FileNotFoundError(f"Density map not found: {density_path}")
        
        density = np.load(density_path).astype('float32')
        return image, density
    
    def train_epoch(self):
        """Train for one epoch"""
        self.regressor.train()
        
        im_ids = self.data_split['train'].copy()
        random.shuffle(im_ids)
        
        train_mae = 0
        train_rmse = 0
        train_loss = 0
        
        pbar = tqdm(im_ids, desc="Training")
        
        for cnt, im_id in enumerate(pbar, 1):
            try:
                # Load annotations
                anno = self.annotations[im_id]
                bboxes = anno['box_examples_coordinates']
                dots = np.array(anno['points'])
                
                # Convert bboxes to rects format [y1, x1, y2, x2]
                rects = []
                for bbox in bboxes:
                    x1, y1 = bbox[0][0], bbox[0][1]
                    x2, y2 = bbox[2][0], bbox[2][1]
                    rects.append([y1, x1, y2, x2])
                
                # Load image and density
                image, density = self._load_image_and_density(im_id)
                
                # Prepare sample
                sample = {
                    'image': image,
                    'lines_boxes': rects,
                    'gt_density': density
                }
                sample = TransformTrain(sample)
                
                image = sample['image'].to(self.device)
                boxes = sample['boxes'].to(self.device)
                gt_density = sample['gt_density'].to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features = extract_features(
                        self.resnet50_conv,
                        image.unsqueeze(0),
                        boxes.unsqueeze(0),
                        MAPS,
                        Scales
                    )
                
                features.requires_grad = True
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.regressor(features)
                
                # Handle size mismatch (when image size isn't divisible by 8)
                if output.shape[2] != gt_density.shape[2] or output.shape[3] != gt_density.shape[3]:
                    orig_count = gt_density.sum().detach().item()
                    gt_density = F.interpolate(
                        gt_density,
                        size=(output.shape[2], output.shape[3]),
                        mode='bilinear'
                    )
                    new_count = gt_density.sum().detach().item()
                    if new_count > 0:
                        gt_density = gt_density * (orig_count / new_count)
                
                # Compute loss
                loss = self.criterion(output, gt_density)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                pred_cnt = torch.sum(output).item()
                gt_cnt = torch.sum(gt_density).item()
                cnt_err = abs(pred_cnt - gt_cnt)
                train_mae += cnt_err
                train_rmse += cnt_err ** 2
                
                # Update progress bar
                pbar.set_description(
                    f"Train | GT: {gt_cnt:6.1f}, Pred: {pred_cnt:6.1f}, "
                    f"Error: {cnt_err:6.1f}, MAE: {train_mae/cnt:5.2f}, "
                    f"RMSE: {(train_rmse/cnt)**0.5:5.2f}"
                )
                
            except Exception as e:
                print(f"\nWarning: Failed to process {im_id}: {str(e)}")
                continue
        
        num_samples = len(im_ids)
        train_loss = train_loss / num_samples if num_samples > 0 else 0
        train_mae = train_mae / num_samples if num_samples > 0 else 0
        train_rmse = (train_rmse / num_samples) ** 0.5 if num_samples > 0 else 0
        
        return train_loss, train_mae, train_rmse
    
    def evaluate(self, split='val'):
        """Evaluate on specified data split"""
        self.regressor.eval()
        
        if split not in self.data_split:
            print(f"Warning: Split '{split}' not found in data")
            return 0, 0
        
        im_ids = self.data_split[split]
        
        sae = 0  # Sum of Absolute Errors
        sse = 0  # Sum of Squared Errors
        cnt = 0
        
        pbar = tqdm(im_ids, desc=f"Evaluating on {split}")
        
        for im_id in pbar:
            try:
                # Load annotations
                anno = self.annotations[im_id]
                bboxes = anno['box_examples_coordinates']
                dots = np.array(anno['points'])
                
                # Convert bboxes to rects format
                rects = []
                for bbox in bboxes:
                    x1, y1 = bbox[0][0], bbox[0][1]
                    x2, y2 = bbox[2][0], bbox[2][1]
                    rects.append([y1, x1, y2, x2])
                
                # Load image (no density needed for inference)
                image_path = join(self.args.image_dir, im_id)
                image = Image.open(image_path)
                image.load()
                
                # Prepare sample
                sample = {'image': image, 'lines_boxes': rects}
                sample = Transform(sample)
                
                image = sample['image'].to(self.device)
                boxes = sample['boxes'].to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    output = self.regressor(
                        extract_features(
                            self.resnet50_conv,
                            image.unsqueeze(0),
                            boxes.unsqueeze(0),
                            MAPS,
                            Scales
                        )
                    )
                
                gt_cnt = dots.shape[0]
                pred_cnt = output.sum().item()
                cnt += 1
                err = abs(gt_cnt - pred_cnt)
                sae += err
                sse += err ** 2
                
                pbar.set_description(
                    f"Eval {split} | GT: {gt_cnt:6d}, Pred: {pred_cnt:6.1f}, "
                    f"Error: {err:6.1f}, MAE: {sae/cnt:5.2f}, RMSE: {(sse/cnt)**0.5:5.2f}"
                )
                
            except Exception as e:
                print(f"\nWarning: Failed to evaluate {im_id}: {str(e)}")
                continue
        
        mae = sae / cnt if cnt > 0 else 0
        rmse = (sse / cnt) ** 0.5 if cnt > 0 else 0
        
        print(f"\n{split.upper()} Results - MAE: {mae:6.2f}, RMSE: {rmse:6.2f}")
        return mae, rmse
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = join(self.args.output_dir, "checkpoints")
        
        if is_best:
            checkpoint_path = join(checkpoint_dir, "best_model.pth")
        else:
            checkpoint_path = join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        
        torch.save(self.regressor.state_dict(), checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    def save_stats(self):
        """Save training statistics to file"""
        stats_file = join(self.args.output_dir, "stats", "finetune_stats.txt")
        
        with open(stats_file, 'w') as f:
            f.write("epoch,train_loss,train_mae,train_rmse,val_mae,val_rmse\n")
            for stats in self.training_stats:
                f.write(",".join([str(x) for x in stats]) + "\n")
        
        print(f"✓ Saved stats to: {stats_file}")
    
    def finetune(self):
        """Main finetuning loop"""
        print("\n" + "="*80)
        print("Starting Finetuning")
        print("="*80)
        
        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            
            # Train
            train_loss, train_mae, train_rmse = self.train_epoch()
            
            # Evaluate periodically
            if (epoch + 1) % self.args.eval_frequency == 0:
                self.regressor.eval()
                val_mae, val_rmse = self.evaluate(self.args.test_split)
                
                # Update learning rate scheduler
                self.scheduler.step(val_mae)
                
                # Track stats
                self.training_stats.append((
                    epoch + 1,
                    train_loss,
                    train_mae,
                    train_rmse,
                    val_mae,
                    val_rmse
                ))
                
                # Save best model
                if val_mae < self.best_mae:
                    self.best_mae = val_mae
                    self.best_rmse = val_rmse
                    self.save_checkpoint(epoch + 1, is_best=True)
                
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Train MAE:  {train_mae:.2f} | Train RMSE: {train_rmse:.2f}")
                print(f"  Val MAE:    {val_mae:.2f} | Val RMSE:   {val_rmse:.2f}")
                print(f"  Best Val MAE: {self.best_mae:.2f} | Best RMSE: {self.best_rmse:.2f}")
            else:
                self.training_stats.append((
                    epoch + 1,
                    train_loss,
                    train_mae,
                    train_rmse,
                    0,
                    0
                ))
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Train MAE:  {train_mae:.2f} | Train RMSE: {train_rmse:.2f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.args.save_frequency == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
            
            # Save stats after each epoch
            self.save_stats()
        
        print("\n" + "="*80)
        print(f"Finetuning Complete!")
        print(f"Best Validation MAE: {self.best_mae:.2f}")
        print(f"Best Validation RMSE: {self.best_rmse:.2f}")
        print(f"Results saved to: {self.args.output_dir}")
        print("="*80 + "\n")


def main():
    args = setup_finetune_parser()
    
    # Set CUDA settings
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Initialize and run finetuning
    finetune_model = FinetunedCountingModel(args)
    finetune_model.finetune()


if __name__ == "__main__":
    main()
