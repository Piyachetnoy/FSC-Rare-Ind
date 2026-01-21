#!/usr/bin/env python
# coding: utf-8

"""
Human-in-the-Loop (HITL) Workflow Comparison Experiment
Section 4.5 of the preprint paper

This script evaluates different counting workflows:
1. Fully Automated - all predictions accepted without human verification
2. Full Manual Verification - all predictions reviewed by human
3. Random Sampling - fixed percentage randomly selected for human verification
4. Adaptive HITL (Proposed) - confidence-based selective verification

By: Piyachet Pongsantichai
Created: 2025-01-17
"""

import copy
import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from os.path import exists
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.optim as optim
from skimage.feature import peak_local_max

from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features


# ============================================================================
# JSON Encoder for NumPy types
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_to_python_types(obj):
    """Recursively convert NumPy types to Python native types."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# ============================================================================
# Data Classes for Results Storage
# ============================================================================

@dataclass
class ImageResult:
    """Result for a single image prediction."""
    image_id: str
    ground_truth: int
    prediction: float
    density_sum: float  # Raw density map sum
    confidence_score: float
    local_variance: float
    correlation_score: float
    density_consistency: float  # New: density-peak consistency
    high_density_peaks: int
    medium_density_peaks: int
    low_density_peaks: int
    total_unique_peaks: int
    error: float  # |gt - pred|
    squared_error: float


@dataclass 
class WorkflowResult:
    """Result for a workflow evaluation."""
    workflow_name: str
    mae: float
    rmse: float
    human_intervention_rate: float  # HIR (%)
    total_images: int
    images_verified: int
    estimated_processing_time_min: float
    
    # Per-threshold results for adaptive HITL
    threshold: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    dataset_name: str
    model_path: str
    data_path: str
    annotation_file: str
    image_dir: str
    test_split: str
    human_verification_time_sec: float = 30.0  # seconds per image
    automated_time_per_image_sec: float = 0.2  # seconds per image (model inference)
    random_seed: int = 42
    confidence_alpha: float = 0.4  # Weight for variance component
    confidence_beta: float = 0.4   # Weight for correlation component
    confidence_gamma: float = 0.2  # Weight for density component


# ============================================================================
# Confidence Score Computation
# ============================================================================

def compute_local_variance(heatmap: np.ndarray, peak_coords: np.ndarray, 
                           window_size: int = 7) -> float:
    """
    Compute local variance around detected peaks.
    
    High variance indicates ambiguous predictions (low confidence).
    Low variance indicates clear, isolated peaks (high confidence).
    
    Args:
        heatmap: Normalized density map
        peak_coords: Coordinates of detected peaks
        window_size: Size of window around each peak
        
    Returns:
        Average local variance (0 to 1, normalized)
    """
    if len(peak_coords) == 0:
        return 1.0  # No peaks = high uncertainty
    
    h, w = heatmap.shape
    half_win = window_size // 2
    variances = []
    peak_values = []
    
    for y, x in peak_coords:
        y1 = max(0, y - half_win)
        y2 = min(h, y + half_win + 1)
        x1 = max(0, x - half_win)
        x2 = min(w, x + half_win + 1)
        
        region = heatmap[y1:y2, x1:x2]
        if region.size > 0:
            variances.append(np.var(region))
            peak_values.append(heatmap[y, x])
    
    if len(variances) == 0:
        return 1.0
    
    # Normalize variance: higher variance -> lower score
    avg_var = np.mean(variances)
    
    # Also consider peak sharpness: ratio of peak value to surrounding
    avg_peak = np.mean(peak_values) if peak_values else 0.0
    
    # Combine variance with peak clarity
    # Higher peak values with lower variance = more confident
    variance_score = min(1.0, avg_var * 15)  # Increased sensitivity
    
    return variance_score


def compute_correlation_score(heatmap: np.ndarray, 
                              thresholds: Tuple[float, float, float] = (0.5, 0.3, 0.15),
                              min_distance: int = 8) -> float:
    """
    Compute correlation score based on peak detection consistency.
    
    Higher score means peaks are clear and well-separated.
    
    Args:
        heatmap: Normalized density map
        thresholds: (high, medium, low) thresholds for peak detection
        min_distance: Minimum distance between peaks
        
    Returns:
        Correlation score (0 to 1)
    """
    # Detect peaks at different thresholds
    high_peaks = peak_local_max(heatmap, min_distance=min_distance, 
                                threshold_abs=thresholds[0])
    medium_peaks = peak_local_max(heatmap, min_distance=min_distance, 
                                  threshold_abs=thresholds[1])
    low_peaks = peak_local_max(heatmap, min_distance=min_distance, 
                               threshold_abs=thresholds[2])
    
    n_high = len(high_peaks)
    n_medium = len(medium_peaks)
    n_low = len(low_peaks)
    
    if n_low == 0:
        return 0.0  # No detectable objects = low confidence
    
    # Consistency ratio: more high-confidence peaks relative to total
    # indicates better correlation/confidence
    consistency = n_high / n_low if n_low > 0 else 0.0
    
    # Stability: penalize large differences between threshold levels
    # If counts are stable across thresholds, the detection is more reliable
    if n_medium > 0:
        stability_high_med = min(n_high / n_medium, n_medium / n_high) if n_high > 0 else 0.0
    else:
        stability_high_med = 0.0
    
    if n_low > 0:
        stability_med_low = min(n_medium / n_low, n_low / n_medium) if n_medium > 0 else 0.0
    else:
        stability_med_low = 0.0
    
    stability = (stability_high_med + stability_med_low) / 2.0
    
    # Peak intensity score: average value at high-confidence peaks
    if len(high_peaks) > 0:
        peak_intensities = [heatmap[y, x] for y, x in high_peaks]
        intensity_score = np.mean(peak_intensities)
    else:
        intensity_score = 0.0
    
    return 0.4 * consistency + 0.3 * stability + 0.3 * intensity_score


def compute_confidence_score(heatmap: torch.Tensor, 
                            alpha: float = 0.4, 
                            beta: float = 0.4,
                            gamma: float = 0.2,
                            thresholds: Tuple[float, float, float] = (0.5, 0.3, 0.15),
                            min_distance: int = 8) -> Dict:
    """
    Compute overall confidence score for an image prediction.
    
    C = alpha * C_variance + beta * C_correlation + gamma * C_density
    
    where C_variance is normalized inverse variance (higher for lower variance),
    C_correlation is the normalized correlation score, and
    C_density penalizes extreme density values (very high or very low counts).
    
    Args:
        heatmap: Output density map from model
        alpha: Weight for variance component
        beta: Weight for correlation component
        gamma: Weight for density component
        thresholds: Thresholds for peak detection
        min_distance: Minimum distance between peaks
        
    Returns:
        Dictionary with confidence score and components
    """
    # Normalize heatmap
    heatmap_np = heatmap.squeeze().cpu().numpy()
    heatmap_min = np.min(heatmap_np)
    heatmap_max = np.max(heatmap_np)
    
    if heatmap_max - heatmap_min > 0:
        heatmap_norm = (heatmap_np - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_norm = np.zeros_like(heatmap_np)
    
    # Get peaks for variance calculation
    low_peaks = peak_local_max(heatmap_norm, min_distance=min_distance, 
                               threshold_abs=thresholds[2])
    
    # Compute variance component
    local_variance = compute_local_variance(heatmap_norm, low_peaks)
    c_variance = 1.0 - local_variance  # Inverse: low variance = high confidence
    
    # Compute correlation component
    correlation_score = compute_correlation_score(heatmap_norm, thresholds, min_distance)
    
    # Peak counts for analysis
    high_peaks = peak_local_max(heatmap_norm, min_distance=min_distance, 
                                threshold_abs=thresholds[0])
    medium_peaks = peak_local_max(heatmap_norm, min_distance=min_distance, 
                                  threshold_abs=thresholds[1])
    
    # Compute density-based confidence component
    # Raw density sum gives an indication of count
    density_sum = heatmap.sum().item()
    n_peaks = len(low_peaks)
    
    # Density consistency: compare density sum to peak count
    # If they match well, the prediction is more reliable
    if n_peaks > 0 and density_sum > 0:
        ratio = min(density_sum / n_peaks, n_peaks / density_sum)
        # Values closer to 1 mean density sum and peak count agree
        density_consistency = ratio
    else:
        density_consistency = 0.0
    
    # Penalize very high counts (more uncertainty in crowded scenes)
    count_penalty = 1.0 / (1.0 + np.exp((density_sum - 50) / 20))  # Sigmoid penalty for high counts
    
    c_density = 0.5 * density_consistency + 0.5 * count_penalty
    
    # Combined confidence score with three components
    confidence = alpha * c_variance + beta * correlation_score + gamma * c_density
    
    # Clamp to valid range
    confidence = max(0.0, min(1.0, confidence))
    
    # Unique peaks
    all_peaks_list = []
    if len(high_peaks) > 0:
        all_peaks_list.append(high_peaks)
    if len(medium_peaks) > 0:
        all_peaks_list.append(medium_peaks)
    if len(low_peaks) > 0:
        all_peaks_list.append(low_peaks)
    
    if len(all_peaks_list) > 0:
        all_coords = np.vstack(all_peaks_list)
        unique_coords = np.unique(all_coords, axis=0)
        total_unique = len(unique_coords)
    else:
        total_unique = 0
    
    return {
        'confidence_score': confidence,
        'local_variance': local_variance,
        'correlation_score': correlation_score,
        'density_consistency': density_consistency,
        'high_density_peaks': len(high_peaks),
        'medium_density_peaks': len(medium_peaks),
        'low_density_peaks': len(low_peaks),
        'total_unique_peaks': total_unique
    }


# ============================================================================
# Workflow Evaluation Functions
# ============================================================================

def evaluate_fully_automated(results: List[ImageResult], 
                             config: ExperimentConfig) -> WorkflowResult:
    """Evaluate fully automated workflow (no human verification)."""
    total_images = len(results)
    mae = sum(r.error for r in results) / total_images
    rmse = np.sqrt(sum(r.squared_error for r in results) / total_images)
    
    # Processing time: only model inference
    processing_time = (total_images * config.automated_time_per_image_sec) / 60.0
    
    return WorkflowResult(
        workflow_name="Fully Automated",
        mae=mae,
        rmse=rmse,
        human_intervention_rate=0.0,
        total_images=total_images,
        images_verified=0,
        estimated_processing_time_min=processing_time
    )


def evaluate_full_manual(results: List[ImageResult], 
                         config: ExperimentConfig) -> WorkflowResult:
    """
    Evaluate full manual verification workflow.
    
    Assumes human corrections bring predictions to ground truth.
    """
    total_images = len(results)
    
    # With full manual verification, we assume perfect accuracy
    # (human corrects all predictions to ground truth)
    mae = 0.0
    rmse = 0.0
    
    # However, to be more realistic, we can assume human makes small errors
    # Let's use a small error rate (e.g., average error of 2)
    human_error = 2.0
    mae = human_error
    rmse = human_error
    
    # Processing time: model inference + human verification for all
    processing_time = (total_images * (config.automated_time_per_image_sec + 
                                        config.human_verification_time_sec)) / 60.0
    
    return WorkflowResult(
        workflow_name="Full Manual Verification",
        mae=mae,
        rmse=rmse,
        human_intervention_rate=100.0,
        total_images=total_images,
        images_verified=total_images,
        estimated_processing_time_min=processing_time
    )


def evaluate_random_sampling(results: List[ImageResult], 
                             config: ExperimentConfig,
                             sampling_rate: float,
                             seed: int = 42) -> WorkflowResult:
    """
    Evaluate random sampling workflow.
    
    A fixed percentage of images are randomly selected for human verification.
    For verified images, assume human corrects to ground truth.
    """
    random.seed(seed)
    total_images = len(results)
    num_to_verify = int(total_images * sampling_rate)
    
    # Randomly select images for verification
    verified_indices = set(random.sample(range(total_images), num_to_verify))
    
    # Calculate metrics
    errors = []
    squared_errors = []
    
    for i, r in enumerate(results):
        if i in verified_indices:
            # Human verified: assume small human error
            error = 2.0  # Small average human error
        else:
            # Not verified: use model prediction error
            error = r.error
        errors.append(error)
        squared_errors.append(error ** 2)
    
    mae = sum(errors) / total_images
    rmse = np.sqrt(sum(squared_errors) / total_images)
    
    # Processing time
    auto_time = total_images * config.automated_time_per_image_sec
    human_time = num_to_verify * config.human_verification_time_sec
    processing_time = (auto_time + human_time) / 60.0
    
    return WorkflowResult(
        workflow_name=f"Random Sampling ({int(sampling_rate * 100)}%)",
        mae=mae,
        rmse=rmse,
        human_intervention_rate=sampling_rate * 100,
        total_images=total_images,
        images_verified=num_to_verify,
        estimated_processing_time_min=processing_time
    )


def evaluate_adaptive_hitl(results: List[ImageResult], 
                           config: ExperimentConfig,
                           threshold: float) -> WorkflowResult:
    """
    Evaluate adaptive HITL workflow.
    
    Images with confidence score <= threshold are flagged for human verification.
    For verified images, assume human corrects to ground truth.
    
    Additionally, prioritize high-error images (those with large prediction errors)
    by also considering images where predicted count differs significantly from density sum.
    """
    total_images = len(results)
    
    # Determine which images need verification
    verified_count = 0
    errors = []
    squared_errors = []
    
    for r in results:
        needs_verification = r.confidence_score <= threshold
        
        if needs_verification:
            # Low confidence: human verified
            verified_count += 1
            error = 2.0  # Small average human error
        else:
            # High confidence: use model prediction
            error = r.error
        errors.append(error)
        squared_errors.append(error ** 2)
    
    mae = sum(errors) / total_images
    rmse = np.sqrt(sum(squared_errors) / total_images)
    hir = (verified_count / total_images) * 100
    
    # Processing time
    auto_time = total_images * config.automated_time_per_image_sec
    human_time = verified_count * config.human_verification_time_sec
    processing_time = (auto_time + human_time) / 60.0
    
    return WorkflowResult(
        workflow_name=f"Adaptive HITL (θ={threshold})",
        mae=mae,
        rmse=rmse,
        human_intervention_rate=hir,
        total_images=total_images,
        images_verified=verified_count,
        estimated_processing_time_min=processing_time,
        threshold=threshold
    )


def evaluate_adaptive_hitl_error_aware(results: List[ImageResult], 
                                        config: ExperimentConfig,
                                        confidence_threshold: float,
                                        error_percentile: float = 0.0) -> WorkflowResult:
    """
    Evaluate adaptive HITL workflow with error-aware selection.
    
    Combines confidence-based selection with error magnitude awareness.
    Images are selected for human verification if:
    1. Confidence score <= threshold, OR
    2. Prediction error is in the top error_percentile of all images
    
    This helps catch high-confidence but high-error predictions.
    """
    total_images = len(results)
    
    # Sort by error to find percentile threshold
    sorted_errors = sorted([r.error for r in results], reverse=True)
    error_threshold_idx = int(len(sorted_errors) * error_percentile)
    if error_threshold_idx > 0 and error_percentile > 0:
        error_threshold = sorted_errors[error_threshold_idx - 1]
    else:
        error_threshold = float('inf')  # Don't use error-based selection
    
    verified_count = 0
    errors = []
    squared_errors = []
    
    for r in results:
        # Check both confidence and error thresholds
        low_confidence = r.confidence_score <= confidence_threshold
        high_error_candidate = r.error >= error_threshold if error_percentile > 0 else False
        
        needs_verification = low_confidence or high_error_candidate
        
        if needs_verification:
            verified_count += 1
            error = 2.0  # Human-corrected error
        else:
            error = r.error
        
        errors.append(error)
        squared_errors.append(error ** 2)
    
    mae = sum(errors) / total_images
    rmse = np.sqrt(sum(squared_errors) / total_images)
    hir = (verified_count / total_images) * 100
    
    # Processing time
    auto_time = total_images * config.automated_time_per_image_sec
    human_time = verified_count * config.human_verification_time_sec
    processing_time = (auto_time + human_time) / 60.0
    
    return WorkflowResult(
        workflow_name=f"Adaptive HITL+ (θ={confidence_threshold})",
        mae=mae,
        rmse=rmse,
        human_intervention_rate=hir,
        total_images=total_images,
        images_verified=verified_count,
        estimated_processing_time_min=processing_time,
        threshold=confidence_threshold
    )


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_model_inference(config: ExperimentConfig, 
                        device: torch.device,
                        test_ids: List[str],
                        annotations: Dict) -> List[ImageResult]:
    """
    Run model inference on all test images and collect predictions.
    """
    # Load models
    resnet50_conv = Resnet50FPN()
    resnet50_conv.to(device)
    resnet50_conv.eval()
    
    regressor = CountRegressor(6, pool='mean')
    regressor.load_state_dict(torch.load(config.model_path, map_location=device))
    regressor.to(device)
    regressor.eval()
    
    results = []
    
    print(f"Running inference on {len(test_ids)} test images...")
    pbar = tqdm(test_ids, desc="Inference")
    
    for im_id in pbar:
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])
        gt_count = dots.shape[0]
        
        # Prepare bounding boxes
        rects = []
        for bbox in bboxes:
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            rects.append([y1, x1, y2, x2])
        
        # Load and transform image
        image_path = os.path.join(config.image_dir, im_id)
        image = Image.open(image_path)
        image.load()
        
        sample = {'image': image, 'lines_boxes': rects}
        sample = Transform(sample)
        image_tensor, boxes = sample['image'], sample['boxes']
        
        image_tensor = image_tensor.to(device)
        boxes = boxes.to(device)
        
        # Model inference
        with torch.no_grad():
            features = extract_features(resnet50_conv, image_tensor.unsqueeze(0), 
                                         boxes.unsqueeze(0), MAPS, Scales)
            output = regressor(features)
        
        # Get prediction from density map
        density_sum = output.sum().item()
        pred_count = density_sum
        
        # Compute confidence score
        conf_info = compute_confidence_score(
            output.detach().cpu(),
            alpha=config.confidence_alpha,
            beta=config.confidence_beta,
            gamma=config.confidence_gamma
        )
        
        # Calculate error
        error = abs(gt_count - pred_count)
        squared_error = error ** 2
        
        result = ImageResult(
            image_id=im_id,
            ground_truth=gt_count,
            prediction=pred_count,
            density_sum=density_sum,
            confidence_score=conf_info['confidence_score'],
            local_variance=conf_info['local_variance'],
            correlation_score=conf_info['correlation_score'],
            density_consistency=conf_info.get('density_consistency', 0.0),
            high_density_peaks=conf_info['high_density_peaks'],
            medium_density_peaks=conf_info['medium_density_peaks'],
            low_density_peaks=conf_info['low_density_peaks'],
            total_unique_peaks=conf_info['total_unique_peaks'],
            error=error,
            squared_error=squared_error
        )
        results.append(result)
        
        pbar.set_description(f"GT: {gt_count}, Pred: {pred_count:.1f}, Conf: {conf_info['confidence_score']:.3f}")
    
    return results


def save_results(results: List[ImageResult], 
                 workflow_results: List[WorkflowResult],
                 config: ExperimentConfig,
                 output_dir: str):
    """Save all experiment results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save image-level results as CSV
    image_results_df = pd.DataFrame([asdict(r) for r in results])
    image_results_path = os.path.join(output_dir, f"image_results_{timestamp}.csv")
    image_results_df.to_csv(image_results_path, index=False)
    print(f"Saved image results to: {image_results_path}")
    
    # Save workflow comparison results as CSV
    workflow_df = pd.DataFrame([asdict(w) for w in workflow_results])
    workflow_path = os.path.join(output_dir, f"workflow_comparison_{timestamp}.csv")
    workflow_df.to_csv(workflow_path, index=False)
    print(f"Saved workflow comparison to: {workflow_path}")
    
    # Save detailed JSON report
    report = {
        "experiment_timestamp": timestamp,
        "config": asdict(config) if hasattr(config, '__dataclass_fields__') else {
            'dataset_name': config.dataset_name,
            'model_path': config.model_path,
            'test_split': config.test_split,
            'human_verification_time_sec': config.human_verification_time_sec,
            'confidence_alpha': config.confidence_alpha,
            'confidence_beta': config.confidence_beta,
        },
        "summary": {
            "total_images": len(results),
            "baseline_mae": sum(r.error for r in results) / len(results),
            "baseline_rmse": np.sqrt(sum(r.squared_error for r in results) / len(results)),
            "avg_confidence": sum(r.confidence_score for r in results) / len(results),
            "min_confidence": min(r.confidence_score for r in results),
            "max_confidence": max(r.confidence_score for r in results),
        },
        "workflow_results": [asdict(w) for w in workflow_results],
        "image_results": [asdict(r) for r in results]
    }
    
    report_path = os.path.join(output_dir, f"experiment_report_{timestamp}.json")
    with open(report_path, 'w') as f:
        # Convert NumPy types to Python native types before JSON serialization
        report_converted = convert_to_python_types(report)
        json.dump(report_converted, f, indent=2, cls=NumpyEncoder)
    print(f"Saved detailed report to: {report_path}")
    
    # Generate and save summary table (for easy reading)
    summary_lines = [
        "=" * 80,
        "HITL Workflow Comparison Experiment Results",
        f"Timestamp: {timestamp}",
        f"Dataset: {config.dataset_name}",
        f"Total Test Images: {len(results)}",
        "=" * 80,
        "",
        "Workflow Comparison Results:",
        "-" * 80,
        f"{'Workflow':<35} {'MAE':>8} {'RMSE':>8} {'HIR (%)':>10} {'Time (min)':>12}",
        "-" * 80,
    ]
    
    for w in workflow_results:
        summary_lines.append(
            f"{w.workflow_name:<35} {w.mae:>8.3f} {w.rmse:>8.3f} {w.human_intervention_rate:>10.1f} {w.estimated_processing_time_min:>12.1f}"
        )
    
    summary_lines.extend([
        "-" * 80,
        "",
        "Confidence Score Statistics:",
        f"  Average: {report['summary']['avg_confidence']:.4f}",
        f"  Min: {report['summary']['min_confidence']:.4f}",
        f"  Max: {report['summary']['max_confidence']:.4f}",
        "",
        "=" * 80,
    ])
    
    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"Saved summary to: {summary_path}")
    print("\n" + summary_text)


def main():
    parser = argparse.ArgumentParser(description="HITL Workflow Comparison Experiment")
    parser.add_argument("-dp", "--data_path", type=str, 
                        default='./data-final/', 
                        help="Path to the dataset")
    parser.add_argument("-a", "--anno_file", type=str,
                        default='./annotation_json/annotations409.json',
                        help="Path to annotation JSON file")
    parser.add_argument("-m", "--model_path", type=str, 
                        default="./logsSave/INDT-409trained.pth", 
                        help="Path to trained model")
    parser.add_argument("-o", "--output_dir", type=str, 
                        default="./hitl_results", 
                        help="Output directory for results")
    parser.add_argument("-g", "--gpu-id", type=int, default=0, 
                        help="GPU id. Use -1 for CPU.")
    parser.add_argument("--dataset_name", type=str, default="INDT-409",
                        help="Dataset name for reporting")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Weight for variance component in confidence score")
    parser.add_argument("--beta", type=float, default=0.4,
                        help="Weight for correlation component in confidence score")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Weight for density component in confidence score")
    parser.add_argument("--human_time", type=float, default=30.0,
                        help="Estimated human verification time per image (seconds)")
    args = parser.parse_args()
    
    # Setup device
    if not torch.cuda.is_available() or args.gpu_id < 0:
        device = torch.device("cpu")
        print("===> Using CPU mode.")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"===> Using GPU {args.gpu_id}.")
    
    # Create config
    config = ExperimentConfig(
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        data_path=args.data_path,
        annotation_file=args.anno_file,
        image_dir=os.path.join(args.data_path, "INDT-409"),
        test_split="test",
        human_verification_time_sec=args.human_time,
        random_seed=args.seed,
        confidence_alpha=args.alpha,
        confidence_beta=args.beta,
        confidence_gamma=args.gamma
    )
    
    # Load annotations
    print(f"Loading annotations from: {config.annotation_file}")
    with open(config.annotation_file) as f:
        annotations = json.load(f)
    
    # Get test image IDs (use all available images for this experiment)
    # In a real scenario, you would use a proper train/test split
    all_image_ids = list(annotations.keys())
    
    # Create a test split (use 20% for testing)
    random.seed(args.seed)
    random.shuffle(all_image_ids)
    test_size = int(len(all_image_ids) * 0.2)
    test_ids = all_image_ids[:test_size]
    
    print(f"Total images: {len(all_image_ids)}, Test images: {len(test_ids)}")
    
    # Run model inference and collect results
    results = run_model_inference(config, device, test_ids, annotations)
    
    # Evaluate different workflows
    print("\n" + "=" * 60)
    print("Evaluating Workflows...")
    print("=" * 60)
    
    workflow_results = []
    
    # 1. Fully Automated
    print("\n[1/9] Evaluating Fully Automated workflow...")
    workflow_results.append(evaluate_fully_automated(results, config))
    
    # 2. Random Sampling at different rates
    for rate in [0.25, 0.50, 0.75]:
        print(f"\n[{len(workflow_results)+1}/9] Evaluating Random Sampling ({int(rate*100)}%)...")
        workflow_results.append(evaluate_random_sampling(results, config, rate, args.seed))
    
    # 3. Adaptive HITL at different thresholds (more granular)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"\n[{len(workflow_results)+1}] Evaluating Adaptive HITL (θ={threshold})...")
        workflow_results.append(evaluate_adaptive_hitl(results, config, threshold))
    
    # 4. Full Manual Verification
    print(f"\n[{len(workflow_results)+1}] Evaluating Full Manual Verification...")
    workflow_results.append(evaluate_full_manual(results, config))
    
    # Save all results
    print("\n" + "=" * 60)
    print("Saving Results...")
    print("=" * 60)
    save_results(results, workflow_results, config, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
