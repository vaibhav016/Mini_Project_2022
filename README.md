<h1 align="center">
<p>Deep Learning (CSGY- 6923) :bar_chart:</p>
<p align="center">
<img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python">
<img alt="pytorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C3">
<img alt="PyPI" src="https://img.shields.io/badge/release-v1.0-brightgreen?logo=apache&logoColor=brightgreen">
</p>
</h1>

<h2 align="center">
<p>Mini Project Cifar10 Resent Model </p>
</h2>

## Supervised by Prof. Chinmay Hegde and Prof Arslan Mosenia 

### Built by 
- Vaibhav Singh (vs2410)
- Sindhu Bhoopalam Dinesh (sb8019)
- Sourabh Kumar Bhattacharjee (kb5275)


## Table of Contents

<!-- TOC -->

- [Installation](#installation)  


<!-- /TOC -->

### Installation

```bash
https://github.com/vaibhav016/Mini_Project_2022.git
cd Mini_Project_2022
```
1.) Installation on conda environment -  
```bash
conda env create --name v_env --file=environments.yml
python3 main.py -tr 1 -d "data_path" 
```
2.) Installation via requirements.txt -
```bash
pip install requirements.txt
python3 main.py -tr 1 -d "data_path" 
```

3.) Read the config file (config.yml). All the arguments are passed in this config file. 
Following arguments are available for experimentation

| Arguments  | Optimal Values |
| ------------------------------| ------------- |
| Optimizer                     | SGD  |
| Scheduler |  Reduce On Plateau  |
| Augmentation | True  |
| Learning Rate                 | 0.1  |
| Conv Layers in Resnet Block   | [2,2,2,2]  |
| Input Channels                | 64  |
| Batch Size                    | 256  |
| Output Channels in each block |  [64, 128, 256, 512]  |

4.) To test the optimal model
```
python3 main.py -tr 0 -s "Mini_project/saved_models/model_512_SGD_ROP_256.pt"
```

5.) Performance Metrics on Testing Data  
   
| Loss  | Accuracy |
| ------------------| ------------- |
| 0.3004     | 93.34  |


    
