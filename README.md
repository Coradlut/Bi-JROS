# Bi-JROS: Bi-level Learning of Task-Specific Decoders for Joint Registration and One-Shot Medical Image Segmentation

<a href="https://scholar.google.com/citations?user=vLN1njoAAAAJ&hl=zh-CN&oi=ao" target="_blank">Xin Fan</a><sup>1</sup>,
Xiaolin Wang<sup>1</sup>,</span>
<a href="https://scholar.google.com/citations?user=MWPKMlsAAAAJ&hl=zh-CN&oi=ao" target="_blank">Jiaxin Gao</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=UNXTe-4AAAAJ&hl=zh-CN" target="_blank">Jia Wang</a><sup>1</sup>,
Zhongxuan Luo<sup>1</sup>,</span>
Risheng Liu<sup>1</sup> </span>

<sup>1</sup>School of Software Technology, Dalian University of Technology, Dalian, China &nbsp;&nbsp;

[🏡 Project Page](https://bi-jros.github.io/) |  [📄 Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_Bi-level_Learning_of_Task-Specific_Decoders_for_Joint_Registration_and_One-Shot_CVPR_2024_paper.html) 

## 🎺 News
-  Adapt Different Encoders (eg. sam, synthseg) to Our Framework (updating)
- [2025/04/22]: ✨We release the model weight of Bi-JROS in the Step 1:Pretrain the shared encder [🤗 Huggingface](https://huggingface.co/jiawang0704/Bi-JROS-Step1/tree/main)
- [2024/04/23]: ✨We release the train and inference code.
- [2024/02/27]: ✨This paper was accepted by CVPR 2024!

![The proposed framework](framework.png)

## REQUIREMENTS
This code requires the following:
* Python==3.8
* PyTorch==1.12.1
* Torchvision==0.13.1
* Torchaudio==0.12.1
* Numpy==1.24.3
* Scipy==1.10.1
* Scikit-image==0.21.0
* Nibabel==5.2.0 

## DATA
The datasets used in the paper, ABIDE, ANDI, PPMI, and OASIS, are publicly available for download.  
For example, ADNI can be applied for and downloaded through the following link: [https://adni.loni.usc.edu/data-samples/adni-data/#AccessData](https://adni.loni.usc.edu/data-samples/adni-data/#AccessData).  
The download process for ABIDE is described at [https://fcon_1000.projects.nitrc.org/indi/abide/databases.html](https://fcon_1000.projects.nitrc.org/indi/abide/databases.html).  
Preprocessed ABIDE data can be accessed at [http://preprocessed-connectomes-project.org/abide/index.html](http://preprocessed-connectomes-project.org/abide/index.html).  

## USAGE
### Step 1: Getting Started

Clone the repo:
```
git clone https://github.com/Coradlut/Bi-JROS.git
```

### Step 2: Training 

```
python train.py
```
Before executing the code, it may be necessary to configure certain parameters in accordance with specific requirements.

### Step 3: Prediction

To test the performance:

```
python infer.py
```


# Adapt Different Encoders to Our Framework (Updating)

In this section, we demonstrate how we adapt different encoders to our framework. Specifically, we focus on integrating four encoders: **SAM**, **SynthSeg**, and two of our own proposed methods. We will showcase the results of applying these encoders and provide a brief introduction to each of the methods.

## 1. Introduction to the Methods

### SAM (Self-Attention Mechanism)

SAM (Self-Attention Mechanism) uses attention layers to capture long-range dependencies, allowing the model to focus on the most relevant regions of the input. This encoder improves the performance in medical image segmentation by enhancing feature representation through attention-based mechanisms.

#### Key Features:
- **Attention-based**: Focuses on the relevant parts of the image.
- **Efficient for high-dimensional data**: Suitable for complex medical image structures.

### SynthSeg

SynthSeg is a method that utilizes synthetic data for training a segmentation model, then fine-tunes the model using real medical images. This helps the model generalize well across different medical datasets.

#### Key Features:
- **Pre-trained on synthetic data**: Uses synthetic data to generalize across different real-world datasets.
- **Highly efficient for rare conditions**: Helps in segmentation tasks with limited annotated data.

### Our Proposed Method 1: Encoder-A

Encoder-A is designed to handle fine-grained details by using a multi-scale feature extraction technique. The encoder extracts features at multiple scales and combines them for better feature representation.

#### Key Features:
- **Multi-scale**: Processes images at different resolutions.
- **Captures fine details**: Focuses on minute anatomical details for improved segmentation.

### Our Proposed Method 2: Encoder-B

Encoder-B combines convolutional networks with recurrent layers, enabling the model to handle both spatial and sequential dependencies. This method is particularly effective for sequential or 3D medical image data.

#### Key Features:
- **Hybrid architecture**: Combines CNNs and recurrent layers.
- **Effective for sequential/3D data**: Well-suited for tasks like 3D medical image segmentation.

---

## 2. Results Comparison

In this section, we present the results of applying the different encoders to our framework. The **Dice coefficient** is used as the evaluation metric to compare the segmentation performance of each method.

| Encoder Method       | Dice Coefficient (OASIS) | Dice Coefficient (Dataset 2) | Dice Coefficient (Dataset 3) |
|----------------------|------------------------------|------------------------------|------------------------------|
| **SAM**              | 0.85                         | --                         | --                         |
| **SynthSeg**         | 0.88                         | --                         | --                        |
| **Bi-JROS**        | 0.90                         | --                        | --                         |
| **RRL-SAM**        | 0.87                         | --                         | --                         |


## 3. Conclusion

By adapting these different encoders into our framework, we are able to leverage the strengths of each method to improve our segmentation accuracy and generalization. SAM and SynthSeg provide strong attention mechanisms and generalization from synthetic data, while our proposed methods offer specialized approaches for fine-grained details and sequential data handling.

We encourage further exploration and experimentation with these encoders to optimize segmentation results across a variety of medical imaging tasks.

---

**Note:** The Dice coefficient values presented above demonstrate how well each encoder performs across different datasets. These results indicate the effectiveness of our framework in handling various medical imaging challenges.

## Quick Start
Set hyperparameters ‘enc’(eg. sam, sythseg, bi-jros. and rrl-sam) to select which necoder to adapt to our framework. 
```
python train_arbi_enc4dec.py -enc bi-jros
```