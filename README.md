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

# Adapt different encoders to our framework
