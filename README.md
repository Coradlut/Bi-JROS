# Bi-JROS: Bi-level Learning of Task-Specific Decoders for Joint Registration and One-Shot Medical Image Segmentation

This is the source code of paper "[Bi-JROS: Bi-level Learning of Task-Specific Decoders for Joint Registration and One-Shot Medical Image Segmentation
]. 

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


