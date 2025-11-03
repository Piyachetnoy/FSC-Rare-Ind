<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]
[![Python][python-shield]][python-url]
[![PyTorch][pytorch-shield]][pytorch-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Few-Shot Counting for Custom Industrial Objects</h3>

  <p align="center">
    Automation of industrial objects counting, using Few-Shot Counting and Feature Detection approach
    <br />
    <a href="https://github.com/Piyachetnoy/rare-ind-counting"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Piyachetnoy/rare-ind-counting/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/Piyachetnoy/rare-ind-counting/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Counting industrial objects is challenging due to their similar appearances and complex shapes. This project adapts **Few-Shot Counting (FSC)** to minimize labeled data requirements while improving accuracy. We use **FamNet** with rule-based feature detection to enhance robustness in industrial settings. Additionally, we introduce the **INDT dataset**, focusing on diverse industrial objects. Our approach integrates density map estimation with feature detection to improve interpretability and reduce over-counting errors.

### Key Features

* **Few-Shot Learning**: Count objects with minimal labeled examples
* **Feature Detection**: Rule-based feature detection for enhanced robustness
* **Density Map Estimation**: Visual density maps for better interpretability
* **Industrial Focus**: Specifically designed for industrial object counting scenarios
* **INDT Dataset**: Custom dataset for diverse industrial objects

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This project is built using the following technologies and frameworks:

* [![Python][python-shield]][python-url] Python 3.10.11
* [![PyTorch][pytorch-shield]][pytorch-url] PyTorch - Deep learning framework
* [![OpenCV][opencv-shield]][opencv-url] OpenCV - Computer vision library
* [![NumPy][numpy-shield]][numpy-url] NumPy - Numerical computing
* [![Pillow][pillow-shield]][pillow-url] Pillow - Image processing

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This section provides instructions on setting up the project locally.

### Prerequisites

Before you begin, ensure you have the following installed:

* Python 3.10.11 or higher
* pip 25.0.1 or higher
* CUDA-capable GPU (recommended for training and inference)
* CUDA toolkit (if using GPU)

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/Piyachetnoy/rare-ind-counting.git
   cd rare-ind-counting
   ```

2. Install Python dependencies
   ```sh
   pip install -r requirements.txt
   ```

3. Download the INDT dataset from [Google Drive](https://drive.google.com/file/d/1TyaHykMSC5rIRx8Js_w58uoOg8kmrt3L/view?usp=sharing) and extract it to the `data/` directory

4. (Optional) Download pre-trained models if available

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Training

Train the model on the INDT dataset:

```sh
python train.py \
  --data_path ./data-V2/ \
  --output_dir ./logsSave \
  --epochs 1500 \
  --gpu 0 \
  --learning-rate 1e-6
```

### Testing

Evaluate the model on test data:

```sh
python test.py \
  --data_path ./data-V2/ \
  --test_split test \
  --model_path ./logsSave/model.pth \
  --gpu-id 0
```

With test-time adaptation:

```sh
python test.py \
  --data_path ./data-V2/ \
  --test_split test \
  --model_path ./logsSave/model.pth \
  --adapt \
  --gradient_steps 100 \
  --learning_rate 1e-7 \
  --gpu-id 0
```

### Demo

Run inference on a single image:

```sh
python demo.py \
  --input-image path/to/image.jpg \
  --bbox-file path/to/bbox.txt \
  --output-dir ./output \
  --model_path ./logsSave/model.pth \
  --gpu-id 0
```

If no bounding box file is provided, the script will prompt you to select bounding boxes interactively.

### Demo with Optimization

Run optimized demo with accuracy level visualization:

```sh
python demoopt.py \
  --input-image path/to/image.jpg \
  --bbox-file path/to/bbox.txt \
  --output-dir ./output \
  --model_path ./logsSave/model.pth \
  --gpu-id 0
```

### Command Line Arguments

#### Training (`train.py`)
- `--data_path` / `-dp`: Path to the dataset directory
- `--output_dir` / `-o`: Output directory for logs and models
- `--test-split` / `-ts`: Data split to use (train/test/val)
- `--epochs` / `-ep`: Number of training epochs (default: 1500)
- `--gpu` / `-g`: GPU ID (default: 0)
- `--learning-rate` / `-lr`: Learning rate (default: 1e-6)

#### Testing (`test.py`)
- `--data_path` / `-dp`: Path to the dataset directory
- `--test_split` / `-ts`: Test split (val_PartA/val_PartB/test_PartA/test_PartB/test/val)
- `--model_path` / `-m`: Path to trained model
- `--adapt` / `-a`: Enable test-time adaptation
- `--gradient_steps` / `-gs`: Number of gradient steps for adaptation (default: 100)
- `--learning_rate` / `-lr`: Learning rate for adaptation (default: 1e-7)
- `--gpu-id` / `-g`: GPU ID (use -1 for CPU)

#### Demo (`demo.py` / `demoopt.py`)
- `--input-image` / `-i`: Path to input image (required)
- `--bbox-file` / `-b`: Path to bounding box file (optional)
- `--output-dir` / `-o`: Output directory (default: current directory)
- `--model_path` / `-m`: Path to trained model
- `--gpu-id` / `-g`: GPU ID (use -1 for CPU)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- DATASET -->
## Dataset

The INDT (Industrial Objects) dataset is a custom dataset designed for few-shot counting of industrial objects. The dataset includes:

- Multiple versions (V1-V5) with different configurations
- Density maps for training
- Annotations in JSON format
- Train/Test/Validation splits

### Download

The dataset can be downloaded from:
[Download INDT Dataset](https://drive.google.com/file/d/1TyaHykMSC5rIRx8Js_w58uoOg8kmrt3L/view?usp=sharing)

### Dataset Structure

```
data-V2/
├── annotations.json          # Image annotations
├── Train_Test_Val.json       # Dataset splits
├── indt-objects-V5/         # Image directory
└── density_map_adaptive_V1/  # Density maps
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- AUTHORS -->
## Authors

**Piyachet Pongsantichai**
- LinkedIn: [@piyachet-p2145](https://www.linkedin.com/in/piyachet-p2145/)
- Email: p.pongsantichai@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.md` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This work builds upon the following research and resources:

* **Learning To Count Everything** (CVPR 2021)
  - Authors: Viresh Ranjan, Udbhav Sharma, Thu Nguyen, Minh Hoai
  - Paper: [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Ranjan_Learning_To_Count_Everything_CVPR_2021_paper.pdf)
* FamNet: Few-Shot Counting Network architecture
* Original implementation by Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)

### Resources

* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [OpenCV Documentation](https://docs.opencv.org/)
* [NumPy Documentation](https://numpy.org/doc/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge
[license-url]: https://github.com/Piyachetnoy/rare-ind-counting/blob/main/LICENSE.md
[python-shield]: https://img.shields.io/badge/Python-3.10.11-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[opencv-shield]: https://img.shields.io/badge/OpenCV-4.10.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white
[opencv-url]: https://opencv.org/
[numpy-shield]: https://img.shields.io/badge/NumPy-1.26.4-013243?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/
[pillow-shield]: https://img.shields.io/badge/Pillow-11.0.0-013243?style=for-the-badge&logo=python&logoColor=white
[pillow-url]: https://python-pillow.org/
