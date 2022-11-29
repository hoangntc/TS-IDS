# Traffic-aware Self-supervised learning for IoT Network Intrusion Detection System

## 1. Directory structure:

```
.
|   README.md
|   environment.yml
|
|--- datasets
|--- model
|--- ouput
|--- src
|   |-- config
|   |   nf_bot_binary.json
|   |   nf_bot_multi.json
|   |   nf_ton_binary.json
|   |   nf_ton_multi.json
|   dataset.py
|   inference.py
|   model.py
|   pipeline.py
|   predict.py
|   train.py
|   trainer.py
|   tuning.py
|   utils.py
```

## 2. Installation

### 2.1 Libraries

To install all neccessary libraries, please run:

```bash
conda env create -f environment.yml
```

In case, the version of Pytorch and Cuda are not compatible on your machine, please remove all related lib in the `.yml` file; then install Pytorch and Pytorch Geometric separately.


### 2.2 PyTorch
Please follow Pytorch installation instruction in this [link](https://pytorch.org/get-started/locally/).


### 2.3 Torch Geometric
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
where `${TORCH}` and `${CUDA}` is version of Pytorch and Cuda.


## 3. Model Architecture

![Model architecture](/figures/framework.png)

### 3.1 Data preparation
Data preparation consists of three steps: split the data into train/val/test, extract the node/edge features and build the graph data.

To prepare data, check the notebook: `experiment/1_preprocess_data.ipynb`

Raw data with the train/val/test indicator can be downloaded [here](https://drive.google.com/drive/folders/1xyXtGvf4DXbM4epIr9JApZgWjIxsAQho?usp=sharing)

### 3.2 Hyper-parameter tuning

```bash
python ../src/tuning.py --name nf_bot_binary >> ../log/nf_bot_binary.txt 2>&1
```

### 3.3 Training and predicting

After having the best hyperparameters, edit config files in `./src/config/`

For training, run:

```bash
python ../src/train.py --name nf_bot_binary
```
The model checkpoints will be stored in `config['checkpoint_dir']`.
For predicting, choose the model checkpoint in the directory of checkpoint, and run:

```bash
python ../src/predict.py --name nf_bot_binary --restore_model_dir restore_model_dir --restore_model_name restore_model_name
```

## 4. Baselines
- XGBoost: `experiment/3_baseline_XGB.ipynb`
- EGraphSAGE: `experiment/3_baseline_EGraphSAGE.ipynb`

## 5. Evaluation
For evaluation and visualization, run the notebook: `experiment/4_evaluation.ipynb`



