# Few-Shot Counting for Custom Industrial Objects

Automation of industrial objects counting, using **Few-Shot Counting** and **Feature Detection** approach.

## Description

Counting industrial objects is challenging due to their similar appearances and complex shapes. This paper
adapts Few-Shot Counting (FSC) to minimize labeled data requirements while improving accuracy. We use Fam-
Net with rule-based feature detection to enhance robustness in industrial settings. Additionally, we introduce the
INDT dataset, focusing on diverse industrial objects. Our approach integrates density map estimation with feature
detection to improve interpretability and reduce over-counting errors. Experimental results show improved accu-
racy on industrial objects and strong generalization to other datasets, highlighting FSC’s potential for industrial
automation, with future work aimed at optimizing model structure and feature extraction for further performance
improvements.

## Getting Started

Clone repository to your local device.<br/>
Our datasets can be found at [Download Datasets](https://drive.google.com/file/d/1TyaHykMSC5rIRx8Js_w58uoOg8kmrt3L/view?usp=sharing)

### Dependencies

* Python 3.10.11
* pip 25.0.1

### Installing

### Executing program

* How to run the program
* Step-by-step bullets
```
pip install -r requirements.txt
```

## Authors

**Piyachet Pongsantichai**  
> p.pongsantichai{at}gmail.com

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [MIT](LICENSE.md) License - see the .md file for details

## Acknowledgments
This work builds upon concepts from:
> Ranjan, V., Sharma, U., Nguyen, T., & Hoai, M. (2021). *Learning To Count Everything*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).  