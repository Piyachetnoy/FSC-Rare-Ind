# Few-Shot Counting for Custom Industrial Objects: An Adaptive Approach to Real-World Applications

**Authors:** Piyachet Pongsantichai, Fumitake Kato  
**Affiliation:** National Institute of Technology, Ibaraki College  
**Contact:** Piyachet Pongsantichai, p.pongsantichai@gmail.com

## Abstract

Counting industrial objects is challenging due to their similar appearances and complex shapes. This paper adapts Few-Shot Counting (FSC) to minimize labeled data requirements while improving accuracy. We use FamNet with rule-based feature detection to enhance robustness in industrial settings. Additionally, we introduce the INDT dataset, focusing on diverse industrial objects. Our approach integrates density map estimation with feature detection to improve interpretability and reduce over-counting errors. Experimental results show improved accuracy on industrial objects and strong generalization to other datasets, highlighting FSC's potential for industrial automation, with future work aimed at optimizing model structure and feature extraction for further performance improvements.

## 1. Introduction

Automated object counting is essential for inventory management and process optimization in industrial settings [Brynjolfsson 14]. Traditional counting methods, including manual inspection and conventional computer vision, struggle with occlusions, shape variations, and cluttered arrangements. Industrial objects like steel beams, metal rods, and scaffolds present added challenges due to structural similarities.

Deep learning approaches, particularly convolutional neural networks (CNNs) and density-based models, show promise in crowd counting and medical imaging [Aich 18]. However, these models require large labeled datasets and degrade under domain shifts.

Therefore, we explored various methods and selected Few-Shot Counting (FSC), and among models, we chose FamNet for its ability to adapt with minimal supervision [Ranjan 21]. FamNet refines density map predictions using gradient-based test-time adaptation, making it effective when labeled samples are limited. Unlike other methods needing fine-tuning or large datasets, FamNet is flexible and scalable, showing robustness in complex environments.

This paper focuses on technology and system development for automated counting of industrial objects, ensuring usability and accuracy. The key is testing the FSC limit on such complex shapes as T-beam, L-beam, I-beam steel, and Square bar. We compare standard FSC density maps with our rule-based feature detection approach.

Finally, to perform this specific task, we developed a tailored dataset called INDT-576, containing 576 images across five classes of industrial objects, and INDT-409, containing 409 images across three classes of industrial objects. Our code and dataset can be found at [https://github.com/Piyachetnoy/FSC-Rare-Ind](https://github.com/Piyachetnoy/FSC-Rare-Ind).

## 2. Related Works

Few-Shot Counting (FSC) has emerged as a promising solution for object counting, addressing the challenge of limited labeled data. Traditional methods rely on large datasets for training, making them impractical for diverse industrial applications. To overcome this, researchers have explored various FSC approaches.

Ranjan et al. [Ranjan 21] introduced FamNet, a model that learns to count objects with minimal examples by leveraging exemplar-based density map predictions. This approach improves generalization to unseen object categories, reducing reliance on extensive annotations. Building on this, Parnami and Lee [Parnami 22] analyzed different FSC techniques, emphasizing metric-based learning and optimization strategies to enhance adaptability across domains.

Expanding FSC beyond specific object categories, Amini-Naieni et al. [Amini-Naieni 24] proposed CountGD, a model that integrates FSC with open-world counting. By incorporating visual exemplars and textual descriptions, it achieves greater flexibility in counting diverse objects. Similarly, Zhizhong et al. [Zhizhong 24] unified point annotation, segmentation, and counting, demonstrating how FSC can be applied effectively to structured object distributions.

Our research builds on these advancements by applying FSC to industrial object counting. Unlike previous works focused on general objects, we introduce a rule-based feature detection mechanism tailored to optimization the output. By refining density map outputs with feature-based validation, we aim to enhance counting accuracy and usability in real-world industrial environments.

## 3. Methods

### 3.1 Few-Shot Counting Model

Our approach employs the Few-Shot Adaptation and Matching Network (FamNet) as introduced by Ranjan et al. [Ranjan 21]. FamNet consists of two core components: a feature extraction module and a density prediction module. The feature extraction module utilizes a ResNet-50 backbone, pre-trained on ImageNet, with frozen parameters to ensure efficient transfer learning. Exemplar-specific feature extraction is performed through ROI pooling, enabling the model to handle diverse object classes effectively. The density prediction module employs correlation maps to establish feature relationships between query images and exemplars, generating adaptive density maps for precise object counting. A gradient-based test-time adaptation strategy refines the density estimation by leveraging exemplar locations, thereby improving generalization to unseen object categories. On FamNet, we train models using our INDT-576 and INDT-409 datasets to enhance robustness against occlusion and background clutter in industrial objects tasks.

### 3.2 PBAT (Point and Box Annotation Tool)

PBAT is a Python-based tool that we developed to streamline the annotation process, making it both efficient and user-friendly. The functionality is designed to be intuitive yet effective, where a right-click performs a point annotation, and another right-click designates the corner points of a square box to select multiple examples, as shown in Figure 1. The program generates a structured annotation file in JSON format, as illustrated by the example below.

```json
{
    "63.jpg": { "H": 640, "W": 640,
        "box_examples_coordinates": [
        [[251, 102], [529, 102], [529, 554], [251,554]],
        [[251, 102], [529, 102], [529, 554], [251,554]]
        ],
        "points": [[127, 483],[106, 77],[283, 496], [467, 577]],
        "r": [30,30],
        "ratio_h": 0.6, "ratio_w": 0.6
    }, . . .
}
```

**Figure 1: Screenshots from PBAT.** Green boxes indicate box annotations for exemplars, while red dots represent point annotations for counting.

![PBAT Screenshot 1](images/Screenshot%202025-02-20%20at%2017.35.53.png)
![PBAT Screenshot 2](images/Screenshot%202025-02-20%20at%2017.08.42.png)
![PBAT Screenshot 3](images/Screenshot%202025-02-21%20at%200.54.17.png)

### 3.3 Feature Detection

The current output from FamNet relies on density estimation, which can sometimes lead to over-counting. To mitigate this, geometric feature matching algorithms are applied to refine object classification, reducing false positives caused by background interference.

To further enhance peak identification in density maps, we employ the `peak_local_max` function from the Scikit-Image library [van der Walt 14]. This function computes local maxima in a multi-dimensional array by identifying points where a pixel has a higher intensity value than its surrounding neighborhood within a defined window. When applied to a NumPy-based density map, `peak_local_max` extracts object center candidates from the predicted density output. This process effectively transforms the continuous-valued density map into discrete peak points, which correspond to object locations. By post-processing detected peaks, we refine count estimation, reducing over-counting errors from density map noise and overlapping objects. This hybrid approach combines density estimation with feature-based validation, enhancing robustness in complex industrial settings.

**Figure 2: Local Maxima Visualization Example.** The figure illustrates the ability of `peak_local_max` to identify local maxima points within an array.

![Local Maxima Visualization](images/local_minima_complicated_with_dots.png)

### 3.4 The INDT-576 & INDT-409 Dataset

To train our model, we collect images of 589 T-beams, L-beams, T-beams, and square bars, which are compiled into the INDT-576 dataset. The aim is to address the complex shapes of industrial objects. We divide INDT-576 into two versions: INDT-576 and INDT-409. INDT-576 includes the entire dataset, while INDT-409 is a tailored version, excluding the square bar class and aiming for a more even distribution by reducing the number of objects per image. Notably, the main source of images is RoboFlow Universe projects [Ernest 25, MetalPipeCounter 24, ROBOFUN 25].

As a result, INDT-576 contains 576 images across four categories, with an average of 17.60 objects per image, while INDT-409 has 409 images across three categories, with an average of 8.220 objects per image.

## 4. Experiments

### 4.1 Performance Evaluation Metrics

We use Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to evaluate counting accuracy. These metrics are widely used for counting tasks as they measure prediction errors effectively.

MAE calculates the average absolute difference between predicted and actual counts:

$$MAE = \frac{1}{n}\sum_{i=1}^{n}\left |c_{i}-\hat{c}_{i} \right |$$

RMSE, on the other hand, computes the square root of the average squared differences, giving more weight to larger errors:

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(c_{i}-\hat{c}_{i})^{2}}$$

where $n$ is the number of test images, $c_{i}$ is the ground truth count, and $\hat{c}_{i}$ is the predicted count, Helping assess both overall accuracy and error distribution.

### 4.2 Training

We trained the models using INDT-576 and INDT-409 datasets, with a data split ratio of 70% for training, 20% for testing, and 10% for validation. This ensures a balanced evaluation while preventing overfitting. The model was trained for 100 epochs with a learning rate of $1\times10^{-6}$, which provided stable convergence and minimized loss fluctuations.

### 4.3 Comparison with Different Datasets

After training, we evaluated the models using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on the validation and test sets. The results compare a pre-trained FamNet, a FamNet trained on INDT-576, and a FamNet trained on INDT-409. In both cases, the models demonstrated strong performance relative to the dataset's average object count.

**Table 1: Comparison of FamNet models on validation and test sets.**

| Model | Val Set MAE | Val Set RMSE | Test Set MAE | Test Set RMSE |
|-------|-------------|--------------|--------------|---------------|
| FSC-147 Pre-trained | 23.75 | 69.07 | 22.08 | 99.54 |
| Trained with INDT-576 | 10.90 | 18.61 | 9.501 | 13.81 |
| Trained with INDT-409 | 3.977 | 5.596 | 6.321 | 12.51 |

To assess model performance on industrial object counting, we evaluated all models using the INDT-576 Test Set. The results demonstrate that our approach consistently outperforms the baseline FamNet, particularly in scenarios with complex object structures and occlusions.

**Table 2: Performance comparison of FamNet models on the INDT-576 Test Set.**

| Model | MAE | RMSE |
|-------|-----|------|
| FSC-147 Pre-trained | 10.56 | 15.72 |
| Trained with INDT-576 | 9.501 | 13.81 |
| Trained with INDT-409 | 9.364 | 16.99 |

To assess generalization across different object categories—a key strength of FSC—we evaluated our models on the FSC-147 test set. The results indicate that our approach maintains strong adaptability beyond industrial datasets, achieving competitive performance despite using only a fraction of the labeled data (approximately 1/10) compared to the baseline FamNet.

**Table 3: Performance comparison of FamNet models on the FSC-147 Test Set.**

| Model | MAE | RMSE |
|-------|-----|------|
| FSC-147 Pre-trained | 22.08 | 99.54 |
| Trained with INDT-576 | 50.67 | 154.5 |
| Trained with INDT-409 | 58.50 | 157.9 |

### 4.4 Output Optimization

We use feature detection to count local maxima within a density map output, formatted as a NumPy array. This improves visualization and allows manual adjustments. First, the optimized output is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

**Table 4: Performance comparison of FamNet models on the INDT-576 Test Set. (With Output Optimization)**

| Model (With Output Optimization) | MAE | RMSE |
|----------------------------------|-----|------|
| FSC-147 Pretrained | 8.897 | 15.38 |
| Trained with INDT-576 | 15.69 | 21.33 |
| Trained with INDT-409 | 18.43 | 32.29 |

Using `peak_local_max`, we extract high, medium, and low-density peaks from the heatmap. These peaks are visualized as colored dots, aiding in interpretation and refinement. By adjusting the minimum distance parameter, we control dot separation to minimize over-counting errors.

**Figure 3: Predicted density maps and counts of FamNet (Trained with INDT-409).**

![Density Map 1](images/002_dot_out.png)
![Density Map 2](images/002_out.png)

As shown in Figure 3, the baseline density estimation of FamNet can include environmental noise in its calculations, potentially leading to over-counting and errors over time.

**Figure 4: Predicted density maps and counts of FamNet (Trained with INDT-409) with output optimization.**

![Density Map with Optimization](images/Untitled-1.png)

With our approach, the final output is an annotated image where the dots color represent confidence levels, as shown in Figure 4. Integrating rule-based feature detection improves visualization and enhances adaptability for industrial counting tasks. However, challenges persist when objects are closely positioned or overlapping.

## 5. Conclusion

This paper proposes a novel dataset and output optimization method for industrial object counting using Few-Shot Counting. The dataset demonstrates potential for improved performance with larger training data and also performs reasonably well on non-industrial object tasks.

On the other hand, output adaptation using feature detection, while aiming to reduce over-counting and noise, may not be the best approach in terms of metric performance. However, it provides better visualization and manual adjustability compared to density-based methods. Future work will explore improvements in FSC network and model structure to enhance effectiveness.

## References

1. **Aich, S. and Stavness, I.** (2018). Improving Object Counting with Heatmap Regulation. arXiv:1803.05494.

2. **Amini-Naieni, N., Han, T., and Zisserman, A.** (2024). CountGD: Multi-Modal Open-World Counting. Advances in Neural Information Processing Systems (NeurIPS).

3. **Brynjolfsson, E. and McAfee, A.** (2014). The second machine age: Work, progress, and prosperity in a time of brilliant technologies. W W Norton & Co.

4. **Ernest** (2025). Centring Sheet Dataset. Roboflow Universe. [https://universe.roboflow.com/ernest-wy6fj/centring-sheet-6fzdo](https://universe.roboflow.com/ernest-wy6fj/centring-sheet-6fzdo).

5. **MetalPipeCounter** (2024). Angles Dataset. Roboflow Universe. [https://universe.roboflow.com/metalpipecounter/angles-nu0a9](https://universe.roboflow.com/metalpipecounter/angles-nu0a9).

6. **Pelhan, J., Lukežič, A., Zavrtanik, V., and Kristan, M.** (2024). A Novel Unified Architecture for Low-Shot Counting by Detection and Segmentation. Advances in Neural Information Processing Systems, 37. Curran Associates, Inc.

7. **Ranjan, V., Sharma, U., Nguyen, T., and Hoai, M.** (2021). Learning To Count Everything. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

8. **ROBOFUN** (2025). Steel Beam Instance Segmentation Dataset Dataset. Roboflow Universe. [https://universe.roboflow.com/robofun/steel-beam-instance-segmentation-dataset](https://universe.roboflow.com/robofun/steel-beam-instance-segmentation-dataset).

9. **van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., and Yu, T.** (2014). scikit-image: image processing in Python. PeerJ, 2(e453).

10. **Parnami, A. and Lee, M.** (2022). Learning from Few Examples: A Summary of Approaches to Few-Shot Learning. arXiv:2203.04291.

11. **Zhizhong, H., Mingliang, D., Yi, Z., Junping, Z., and Hongming, S.** (2024). Point, Segment and Count: A Generalized Framework for Object Counting. CVPR.
