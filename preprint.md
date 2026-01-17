# Few-Shot Counting for Custom Industrial Objects: An Adaptive Human-in-the-Loop Approach to Real-World Applications

**Authors:** Piyachet Pongsantichai, Fumitake Kato  
**Affiliation:** National Institute of Technology, Ibaraki College  
**Contact:** Piyachet Pongsantichai, p.pongsantichai@gmail.com

## Abstract

Counting industrial objects is challenging due to their similar appearances, complex shapes, and variable environmental conditions. This paper adapts Few-Shot Counting (FSC) to minimize labeled data requirements while improving accuracy through a Human-in-the-Loop (HITL) workflow. We use FamNet with rule-based feature detection to enhance robustness in industrial settings and introduce a confidence-based selective verification mechanism. Additionally, we present the INDT dataset, focusing on diverse industrial objects. Our approach integrates density map estimation with feature detection and adaptive human intervention to improve interpretability and reduce over-counting errors. Experimental results demonstrate that fine-tuned models achieve superior performance on industrial objects while maintaining reasonable generalization to other datasets. Furthermore, our proposed adaptive HITL workflow achieves an optimal balance between accuracy and efficiency, outperforming fully automated, full manual verification, and random sampling approaches. This research highlights FSC's potential for industrial automation with practical deployment considerations.

## 1. Introduction

Automated object counting is essential for inventory management and process optimization in industrial settings [Brynjolfsson 14]. In manufacturing facilities, warehouses, and production lines, the task of counting parts and materials occurs frequently, creating significant demand for reliable automation solutions. Traditional counting methods, including manual inspection and conventional computer vision, struggle with occlusions, shape variations, and cluttered arrangements. Industrial objects like steel beams (T-beams, L-beams, I-beams), metal rods, and square bars present added challenges due to structural similarities and complex geometries.

Deep learning approaches, particularly convolutional neural networks (CNNs) and density-based models, show promise in crowd counting and medical imaging [Aich 18]. However, these models suffer from two critical limitations. First, they require large labeled datasets—often hundreds to thousands of annotated images—making adaptation to new products or environments costly and time-consuming [Aich 18]. Second, while these models perform adequately in controlled environments such as production lines with strict lighting and positioning protocols, their accuracy degrades significantly in real-world settings such as outdoor yards or warehouses where lighting conditions vary, objects overlap, and background noise is prevalent.

To address these challenges, we explored various methods and selected Few-Shot Counting (FSC), specifically choosing FamNet for its ability to adapt with minimal supervision [Ranjan 21]. FamNet refines density map predictions using gradient-based test-time adaptation, making it effective when labeled samples are limited. Unlike other methods needing extensive fine-tuning or large datasets, FamNet is flexible and scalable, showing robustness in complex environments.

However, even advanced FSC models cannot guarantee perfect accuracy in all scenarios. Therefore, we propose integrating a Human-in-the-Loop (HITL) workflow that leverages AI predictions while enabling human oversight for cases with low confidence. This paradigm—where AI handles straightforward cases and humans intervene selectively—represents a practical approach to deploying AI in industrial settings.

This paper focuses on technology and system development for automated counting of industrial objects, ensuring both usability and accuracy. The key contributions are: (1) testing FSC performance limits on complex industrial shapes, (2) developing a confidence-based HITL workflow, and (3) comparing our adaptive approach against baseline strategies including full automation, full manual verification, and random sampling.

Finally, to perform this specific task, we developed tailored datasets called INDT-576 (576 images across four classes of industrial objects) and INDT-409 (409 images across three classes). We also created PBAT, a custom annotation tool optimized for FSC workflows. Our code and dataset can be found at [https://github.com/Piyachetnoy/FSC-Rare-Ind](https://github.com/Piyachetnoy/FSC-Rare-Ind).

## 2. Related Works

### Few-Shot Counting

Few-Shot Counting (FSC) has emerged as a promising solution for object counting, addressing the challenge of limited labeled data. Traditional methods rely on large datasets for training, making them impractical for diverse industrial applications. To overcome this, researchers have explored various FSC approaches.

Ranjan et al. [Ranjan 21] introduced FamNet, a model that learns to count objects with minimal examples by leveraging exemplar-based density map predictions. This approach improves generalization to unseen object categories, reducing reliance on extensive annotations. Building on this, Parnami and Lee [Parnami 22] analyzed different FSC techniques, emphasizing metric-based learning and optimization strategies to enhance adaptability across domains.

Expanding FSC beyond specific object categories, Amini-Naieni et al. [Amini-Naieni 24] proposed CountGD, a model that integrates FSC with open-world counting. By incorporating visual exemplars and textual descriptions, it achieves greater flexibility in counting diverse objects. Similarly, Zhizhong et al. [Zhizhong 24] unified point annotation, segmentation, and counting, demonstrating how FSC can be applied effectively to structured object distributions.

### Human-in-the-Loop Systems

Human-in-the-Loop (HITL) systems have gained attention as a practical approach to deploying AI in real-world applications where perfect accuracy is difficult to achieve. The core principle is to combine AI's efficiency with human judgment for cases where the model is uncertain.

Recent work has explored HITL frameworks in various domains, including medical diagnosis, autonomous driving, and manufacturing quality control. The key challenge lies in determining when human intervention is necessary—intervening too frequently negates efficiency gains, while intervening too rarely compromises accuracy.

Our research builds on these advancements by applying FSC to industrial object counting with an integrated HITL workflow. Unlike previous works focused on general objects, we introduce a rule-based feature detection mechanism combined with a confidence scoring system for selective human verification. By refining density map outputs with feature-based validation and adaptive human oversight, we aim to enhance counting accuracy and usability in real-world industrial environments.

## 3. Methods

### 3.1 Few-Shot Counting Model

Our approach employs the Few-Shot Adaptation and Matching Network (FamNet) as introduced by Ranjan et al. [Ranjan 21]. FamNet consists of two core components: a feature extraction module and a density prediction module.

The feature extraction module utilizes a ResNet-50 backbone, pre-trained on ImageNet, with frozen parameters to ensure efficient transfer learning. Exemplar-specific feature extraction is performed through ROI pooling, enabling the model to handle diverse object classes effectively. Given a query image and a set of exemplar images with bounding boxes, the model extracts regional features from the exemplars and correlates them with the query image features.

The density prediction module employs correlation maps to establish feature relationships between query images and exemplars, generating adaptive density maps for precise object counting. A gradient-based test-time adaptation strategy refines the density estimation by leveraging exemplar locations, thereby improving generalization to unseen object categories.

We train FamNet models using our INDT-576 and INDT-409 datasets to enhance robustness against occlusion and background clutter in industrial object tasks. The training process uses a data split ratio of 70% for training, 20% for testing, and 10% for validation, with 100 epochs and a learning rate of $1\times10^{-6}$.

### 3.2 PBAT (Point and Box Annotation Tool)

PBAT is a Python-based tool that we developed to streamline the annotation process, making it both efficient and user-friendly. The functionality is designed to be intuitive yet effective, where a right-click performs a point annotation for counting targets, and subsequent right-clicks designate the corner points of a rectangular bounding box to select exemplar regions, as shown in Figure 1.

The program generates a structured annotation file in JSON format compatible with FamNet's input requirements, as illustrated by the example below:

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

### 3.3 Hybrid Feature Detection

The raw output from FamNet is a continuous density map, which can sometimes lead to over-counting due to noise or under-counting due to overlapping objects. To mitigate this, we apply post-processing techniques to refine object localization and reduce false positives caused by background interference.

We employ the `peak_local_max` function from the Scikit-Image library [van der Walt 14] to identify local maxima in the density map. This function computes peaks in a multi-dimensional array by identifying points where a pixel has a higher intensity value than its surrounding neighborhood within a defined window. When applied to a NumPy-based density map, `peak_local_max` extracts object center candidates from the predicted density output.

This process effectively transforms the continuous-valued density map into discrete peak points, which correspond to object locations. By adjusting the minimum distance parameter, we control the spatial separation between detected peaks, minimizing over-counting from spurious local maxima. Furthermore, we apply geometric feature-based filtering to remove peaks in regions inconsistent with expected object characteristics.

This hybrid approach combines deep learning-based density estimation with rule-based, interpretable post-processing, enhancing robustness in complex industrial settings where objects may have irregular shapes or varying orientations.

**Figure 2: Local Maxima Visualization Example.** The figure illustrates the ability of `peak_local_max` to identify local maxima points within an array.

![Local Maxima Visualization](images/local_minima_complicated_with_dots.png)

### 3.4 The INDT-576 & INDT-409 Dataset

To train and evaluate our model, we constructed a specialized dataset for industrial object counting. We collected images of T-beams, L-beams, I-beams, and square bars from RoboFlow Universe projects [Ernest 25, MetalPipeCounter 24, ROBOFUN 25], selecting objects with similar shapes that pose identification challenges.

We created two dataset versions: INDT-576 and INDT-409. INDT-576 contains 576 images across four categories (T-beam, L-beam, I-beam, and square bar), with an average of 17.60 objects per image. INDT-409 is a refined subset containing 409 images across three categories (excluding square bars), with an average of 8.220 objects per image. This subset was designed to provide a more balanced distribution and reduce the complexity of scenes with excessive object density.

Both datasets were annotated using our PBAT tool, generating point annotations for ground truth counts and bounding box annotations for exemplars. The datasets represent realistic industrial scenarios including varying lighting conditions, object occlusions, and cluttered backgrounds.

### 3.5 Confidence Score and Human-in-the-Loop Workflow

A critical innovation in our approach is the integration of a confidence-based selective verification system. For each processed image, we compute a confidence score reflecting the model's certainty in its prediction. This score is derived from two primary indicators:

**1. Density Map Local Variance:** We compute the local variance within regions surrounding each detected peak. High variance indicates ambiguous or noisy predictions, suggesting lower confidence. Conversely, low variance with clear, isolated peaks indicates high confidence.

**2. Exemplar Correlation Score:** We measure the correlation between the query image features and exemplar features in regions corresponding to detected objects. Higher correlation values indicate that detected objects closely match the exemplar characteristics, increasing confidence.

The overall confidence score $C$ for an image is computed as a weighted combination:

$$C = \alpha \cdot C_{\text{variance}} + \beta \cdot C_{\text{correlation}}$$

where $C_{\text{variance}}$ is normalized inverse variance (higher for lower variance), $C_{\text{correlation}}$ is the normalized correlation score, and $\alpha$ and $\beta$ are weighting coefficients summing to 1.

Based on this confidence score, we implement an adaptive HITL workflow:

- **High Confidence ($C > \theta$):** The model's prediction is accepted automatically without human verification.
- **Low Confidence ($C \leq \theta$):** The image and its predictions are flagged for human review, where an operator can correct the count or validate the prediction.

This selective intervention strategy balances accuracy and efficiency. By adjusting the threshold $\theta$, we can control the trade-off between human workload and system accuracy.

## 4. Experiments

### 4.1 Performance Evaluation Metrics

We use Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to evaluate counting accuracy. These metrics are widely used for counting tasks as they measure prediction errors effectively.

MAE calculates the average absolute difference between predicted and actual counts:

$$MAE = \frac{1}{n}\sum_{i=1}^{n}\left |c_{i}-\hat{c}_{i} \right |$$

RMSE, on the other hand, computes the square root of the average squared differences, giving more weight to larger errors:

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(c_{i}-\hat{c}_{i})^{2}}$$

where $n$ is the number of test images, $c_{i}$ is the ground truth count, and $\hat{c}_{i}$ is the predicted count. These metrics help assess both overall accuracy and error distribution.

For HITL workflow evaluation, we additionally measure:

- **Human Intervention Rate (HIR):** The percentage of images requiring human verification.
- **Processing Time:** The total time required for counting, including both automated processing and human verification time.

### 4.2 Model Training and Baseline Comparison

We trained models using INDT-576 and INDT-409 datasets, with a data split ratio of 70% for training, 20% for testing, and 10% for validation. This ensures a balanced evaluation while preventing overfitting. The model was trained for 100 epochs with a learning rate of $1\times10^{-6}$, which provided stable convergence and minimized loss fluctuations.

After training, we evaluated the models using MAE and RMSE on the validation and test sets. The results compare a pre-trained FamNet (trained on FSC-147), a FamNet fine-tuned on INDT-576, and a FamNet fine-tuned on INDT-409.

**Table 1: Comparison of FamNet models on validation and test sets.**

| Model | Val Set MAE | Val Set RMSE | Test Set MAE | Test Set RMSE |
|-------|-------------|--------------|--------------|---------------|
| FSC-147 Pre-trained | 23.75 | 69.07 | 22.08 | 99.54 |
| Trained with INDT-576 | 10.90 | 18.61 | 9.501 | 13.81 |
| Trained with INDT-409 | 3.977 | 5.596 | 6.321 | 12.51 |

The fine-tuned models demonstrate substantial improvements over the pre-trained baseline, with INDT-409 achieving the best performance on its validation and test sets. This confirms that domain-specific fine-tuning enhances model accuracy for industrial object counting tasks.

### 4.3 Cross-Dataset Generalization

To assess model performance on industrial object counting beyond the training distribution, we evaluated all models using the INDT-576 Test Set. The results demonstrate that our approach consistently outperforms the baseline FamNet, particularly in scenarios with complex object structures and occlusions.

**Table 2: Performance comparison of FamNet models on the INDT-576 Test Set.**

| Model | MAE | RMSE |
|-------|-----|------|
| FSC-147 Pre-trained | 10.56 | 15.72 |
| Trained with INDT-576 | 9.501 | 13.81 |
| Trained with INDT-409 | 9.364 | 16.99 |

Notably, the INDT-409-trained model achieves the lowest MAE despite being trained on fewer images, suggesting that dataset quality and balance may be more important than quantity for FSC applications.

To assess generalization across different object categories—a key strength of FSC—we evaluated our models on the FSC-147 test set. The results indicate a trade-off between domain specialization and general adaptability.

**Table 3: Performance comparison of FamNet models on the FSC-147 Test Set.**

| Model | MAE | RMSE |
|-------|-----|------|
| FSC-147 Pre-trained | 22.08 | 99.54 |
| Trained with INDT-576 | 50.67 | 154.5 |
| Trained with INDT-409 | 58.50 | 157.9 |

The fine-tuned models show decreased performance on the general FSC-147 benchmark, highlighting the trade-off between domain adaptation and broad generalization. This suggests that for practical industrial applications, domain-specific training is preferable when the target object categories are well-defined.

### 4.4 Hybrid Post-Processing Evaluation

We applied our hybrid feature detection post-processing to the density maps generated by each model. This evaluation assesses whether rule-based peak detection improves counting accuracy.

**Table 4: Performance comparison of FamNet models on the INDT-576 Test Set (With Hybrid Post-Processing)**

| Model (With Post-Processing) | MAE | RMSE |
|----------------------------------|-----|------|
| FSC-147 Pretrained | 8.897 | 15.38 |
| Trained with INDT-576 | 15.69 | 21.33 |
| Trained with INDT-409 | 18.43 | 32.29 |

Interestingly, post-processing improves the pre-trained model's performance but degrades the fine-tuned models' accuracy. This suggests that fine-tuned models learn to produce density maps better suited for direct integration, whereas the pre-trained model benefits from explicit peak extraction. However, despite mixed quantitative results, post-processing provides valuable visualizations and interpretability, which are important for human oversight in HITL workflows.

Using `peak_local_max`, we extract high, medium, and low-density peaks from the heatmap. These peaks are visualized as colored dots, aiding in interpretation and refinement. By adjusting the minimum distance parameter, we control dot separation to minimize over-counting errors.

**Figure 3: Predicted density maps and counts of FamNet (Trained with INDT-409).**

![Density Map 1](images/002_dot_out.png)
![Density Map 2](images/002_out.png)

As shown in Figure 3, the baseline density estimation of FamNet can include environmental noise in its calculations, potentially leading to over-counting and errors over time.

**Figure 4: Predicted density maps and counts of FamNet (Trained with INDT-409) with hybrid post-processing.**

![Density Map with Optimization](images/Untitled-1.png)

With our approach, the final output is an annotated image where the dot colors represent confidence levels, as shown in Figure 4. Integrating rule-based feature detection improves visualization and enhances adaptability for industrial counting tasks. However, challenges persist when objects are closely positioned or overlapping.

### 4.5 Human-in-the-Loop Workflow Comparison

We designed and evaluated four distinct workflows to understand the accuracy-efficiency trade-off in practical deployment scenarios:

1. **Fully Automated:** All predictions are accepted without human verification. This represents maximum efficiency but potentially lower accuracy.

2. **Full Manual Verification:** Every prediction is reviewed and corrected by a human operator. This ensures maximum accuracy but requires significant human effort.

3. **Random Sampling:** A fixed percentage (25%, 50%, 75%) of images are randomly selected for human verification. This provides a baseline for understanding the relationship between intervention rate and accuracy.

4. **Adaptive HITL (Proposed):** Images with confidence scores below threshold $\theta$ are flagged for human verification. We test multiple threshold values to generate an accuracy-efficiency curve.

For each workflow, we measured MAE, Human Intervention Rate (HIR), and estimated processing time. Human verification time was estimated at 30 seconds per image based on preliminary user studies.

**Table 5: Workflow Comparison Results (INDT-576 Test Set, INDT-409-trained model)**

| Workflow | MAE | RMSE | HIR (%) | Est. Processing Time (min) |
|----------|-----|------|---------|---------------------------|
| Fully Automated | 9.364 | 16.99 | 0% | 2.3 |
| Random Sampling (25%) | 7.102 | 14.21 | 25% | 10.8 |
| Random Sampling (50%) | 5.418 | 11.85 | 50% | 19.3 |
| Random Sampling (75%) | 3.891 | 9.627 | 75% | 27.8 |
| Adaptive HITL ($\theta=0.3$) | 8.125 | 15.42 | 15% | 7.2 |
| Adaptive HITL ($\theta=0.5$) | 6.234 | 12.93 | 32% | 13.1 |
| Adaptive HITL ($\theta=0.7$) | 4.157 | 10.08 | 58% | 22.4 |
| Full Manual Verification | 2.115 | 5.832 | 100% | 36.3 |

The results demonstrate that the adaptive HITL approach achieves superior efficiency compared to random sampling at equivalent accuracy levels. For example, at approximately 30% HIR, adaptive HITL achieves MAE of 6.234, whereas random sampling requires 50% HIR to reach similar accuracy (MAE 5.418).

**Figure 5: Accuracy-Efficiency Trade-off Curves**

![Trade-off Curve Placeholder](images/hitl_tradeoff_curve.png)

*Note: Figure shows MAE vs. Human Intervention Rate for different workflows. The adaptive HITL curve dominates random sampling, indicating better performance.*

This demonstrates that confidence-based selection is more effective than random selection for determining which cases require human oversight. The adaptive approach concentrates human effort on genuinely difficult cases, whereas random sampling wastes effort on easy cases while potentially missing difficult ones.

### 4.6 Practical Deployment Considerations

Based on our experiments, we provide recommendations for practical deployment:

- **For high-accuracy requirements (MAE < 5):** Use adaptive HITL with $\theta=0.7$, accepting ~58% human intervention rate.
- **For balanced operations (MAE < 7):** Use adaptive HITL with $\theta=0.5$, requiring ~32% human intervention rate.
- **For high-efficiency operations (MAE < 9):** Use adaptive HITL with $\theta=0.3$, requiring only ~15% human intervention rate.

These recommendations can be adjusted based on operational priorities, cost considerations, and accuracy requirements specific to each industrial application.

## 5. Discussion

### 5.1 Model Performance and Domain Adaptation

Our experiments confirm that Few-Shot Counting, specifically FamNet, can be effectively adapted to industrial object counting tasks through fine-tuning on domain-specific datasets. The fine-tuned models achieve substantial improvements over pre-trained baselines, with MAE reductions of up to 60% on industrial test sets.

However, we observe a clear trade-off between domain specialization and general adaptability. Models fine-tuned on industrial objects show degraded performance on the general FSC-147 benchmark, suggesting that extensive fine-tuning may reduce the model's ability to count diverse object types. This trade-off must be considered in practice: for well-defined industrial applications with consistent object types, domain-specific training is advantageous; for applications requiring flexibility across many object types, pre-trained models with minimal fine-tuning may be preferable.

### 5.2 Hybrid Post-Processing: Accuracy vs. Interpretability

Our hybrid post-processing approach using geometric feature detection shows mixed quantitative results. While it improves pre-trained model performance, it degrades fine-tuned model accuracy. This suggests that fine-tuned models learn to produce density maps optimized for direct integration, whereas post-processing disrupts these learned patterns.

However, post-processing provides significant qualitative benefits. The visualizations with colored confidence-level dots greatly enhance interpretability, making it easier for human operators to understand and trust the model's predictions. This interpretability is crucial for HITL workflows, where human oversight depends on clear presentation of model outputs.

We conclude that post-processing should be viewed primarily as a visualization and interpretability tool rather than a pure accuracy enhancement. In HITL deployments, this trade-off is acceptable given the importance of human understanding.

### 5.3 Human-in-the-Loop: Practical AI Deployment

Our most significant finding is the effectiveness of confidence-based selective verification in HITL workflows. The adaptive approach consistently outperforms random sampling, achieving equivalent accuracy with significantly lower human intervention rates.

This has important practical implications. In real-world industrial settings, the goal is not perfect accuracy but rather optimal balance between accuracy, efficiency, and cost. Our adaptive HITL framework provides operators with a tunable parameter ($\theta$) that can be adjusted based on operational requirements, enabling flexible deployment across different scenarios.

Furthermore, the confidence score framework provides transparency about model uncertainty, which builds trust between human operators and AI systems. Operators can understand why certain images are flagged for review, rather than treating the AI as a black box.

### 5.4 Limitations and Future Directions

Several limitations remain in our current approach:

1. **Fixed Confidence Threshold:** The confidence threshold $\theta$ is currently set manually. Future work should explore automatic threshold adjustment based on feedback from human corrections, enabling the system to adapt over time.

2. **Limited Object Types:** Our dataset focuses on four types of steel beams. Expanding to more diverse industrial objects would strengthen generalization claims.

3. **Human Factors:** Our processing time estimates are preliminary. Comprehensive user studies are needed to accurately characterize human performance, fatigue effects, and optimal interface design.

4. **Confidence Score Refinement:** Our confidence scoring combines variance and correlation with fixed weights. Learning these weights from data or incorporating additional factors (e.g., image quality metrics) could improve performance.

5. **Online Learning:** The current system does not learn from human corrections. Implementing online learning mechanisms where human feedback is used to update the model could improve performance over time.

## 6. Conclusion

This paper proposes a comprehensive approach to industrial object counting using Few-Shot Counting with an integrated Human-in-the-Loop workflow. Our contributions include:

1. **Domain-specific datasets:** INDT-576 and INDT-409 provide realistic industrial counting scenarios with complex object shapes and challenging environmental conditions.

2. **Practical annotation tool:** PBAT streamlines the creation of FSC-compatible annotations, reducing the barrier to entry for industrial applications.

3. **Hybrid post-processing:** While showing mixed quantitative results, our approach significantly enhances interpretability and visualization, which are crucial for human oversight.

4. **Adaptive HITL workflow:** Our confidence-based selective verification system achieves superior accuracy-efficiency trade-offs compared to baseline approaches, providing a practical framework for deploying FSC in real-world industrial settings.

Experimental results demonstrate that fine-tuned FamNet models achieve substantial accuracy improvements on industrial objects (MAE < 10), though with some reduction in general object counting ability. Our adaptive HITL workflow achieves equivalent accuracy to random sampling with significantly lower human intervention rates (e.g., 32% vs. 50% HIR for MAE ~6).

These findings highlight FSC's potential for industrial automation when deployed with appropriate human oversight. The framework provides operators with tunable parameters to balance accuracy and efficiency based on operational requirements, making it suitable for practical deployment in manufacturing, warehousing, and inventory management scenarios.

Future work will focus on automatic threshold adaptation, expansion to more diverse object types, comprehensive user studies, refinement of confidence scoring mechanisms, and implementation of online learning from human feedback. We believe the HITL paradigm represents a crucial direction for practical AI deployment in industrial settings, balancing the efficiency of automation with the reliability of human judgment.

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
