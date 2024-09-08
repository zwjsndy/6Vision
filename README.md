# 6Vision

**6Vision: Image-encoding-based IPv6 Target Generation in Few-seed Scenarios**

This model aims to address the detection problem in few-seed scenarios. It requires that the number of seed addresses under each BGP be as few as possible, for example, less than 10. The model is capable of enriching the number of seed addresses under these BGPs, achieving the transformation of few-seed scenarios. However, the algorithm can enter a large number of seed addresses, but because of the use of deep learning, it may take a long time.

What's more, we **collected a number of addresses** in a few-seed scenarios and saved them in `IPv6hitlist_patch.zip`.

<img width="1336" alt="image" src="https://github.com/user-attachments/assets/f20e4300-ac14-4f00-b9cd-95802ac02d9e">



## Environmental requirement

- pytorch 1.12.0 (GPU)
- torchvision 0.13.0
- Python 3.8.18
- ZMapv6
- Others：ipaddress,os,glob,matplotlib,subprocess,time,collections,re,random,numpy,seaborn,sklearn,pickle,pandas,argparse
          (can be installed using pip or conda)

## Running Instructions

### Inputs

The input is in dictionary format (.pkl), where each element in the dictionary follows this structure: the Key is the BGP prefix, and the Value is a list of seed addresses.

For example:
  ```
  {2a01:1111::/32:[2a01:1111::1,2a01:1111::2,2a01:1111::3],
  2a01:1112::/32:[2a01:1112::1,2a01:1112::2,2a01:1112::3],
  2a01:1113::/32:[2a01:1113::1,2a01:1113::2,2a01:1113::3]}
```

### Steps Translation

1. **Cluster** (Default to 6 categories)

   
   Run `cluster.py`, and you will obtain the clustering results in the `label.txt` file.

2. **Model Training**（Default to train 6 independent models and they do not interfere with each other）
   
   
   Run `Gatedpixcelcnn.py`，and you will get 6 models stored in the `./model` directory.

3. **Address Generation**
   
   Run `gen.py` with a parameter `--num`, the value of which can be an integer from 0 to 5, indicating the use of models 0 to 5 to generate candidate addresses.
   The parameter `--budget` allows you to select the size of the budget for each model

   It is recommended to run all, that is, to parallelize these 6 programs:
   ```
   python3 gen.py --num 0 --budget 10000
   python3 gen.py --num 1 --budget 10000
   python3 gen.py --num 2 --budget 10000
   python3 gen.py --num 3 --budget 10000
   python3 gen.py --num 4 --budget 10000
   python3 gen.py --num 5 --budget 10000
   ```
5. **Address probing and Aliased-detection**

   Run `aliasedetect.py`. This step requires a machine with ZMapv6 installed.
   
   You need to enter a parameter `--source_ip`, which is your host machine's IP.

6. **Fine-tuning each Model**
   Run these 6 programs in parallel:
   ```
   python3 retrain.py --num 0
   python3 retrain.py --num 1
   python3 retrain.py --num 2
   python3 retrain.py --num 3
   python3 retrain.py --num 4
   python3 retrain.py --num 5
   ```
7. **Aggregating All Addresses Collected in This Round**
   ```
   python3 alldata.py
   ```
8. **Fine-tune the model**

   To perform fine-tuning for each round, just run steps 4,5 and 6 again. There is no need to execute any other steps. You can repeat the fine-tuning process multiple times based on the need for the number of generated addresses.

   
