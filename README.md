# UT Campus Object Dataset (CODa)

<b>Official dataset development kit for CODa. We strongly recommend using this document to visualize the
dataset and understand its contents.</b>

![Sequence 0 Clip](./docs/CODaComp1000Trim.gif)

## Table of Contents

- <b>[INSTALL](#INSTALL)</b>
- <b>[GETTING STARTED](#GETTING_STARTED)</b>
- <b>[DATA_REPORT](#DATA_REPORT)</b>
- <b>[MODELS](#MODELS)</b>

## <a name="Install"></a>Install

Run the following command to download the development kit. 

```git clone git@github.com:ut-amrl/coda-devkit.git```

We use conda for all package management in this repository. To install all library dependencies, create the 
conda environment using the following command. 

```conda env create -f environment.yml```

Then activate the environment as follows. Further library usage instructions are documented in 
the <b>[GETTING STARTED](#GETTING_STARTED)</b> section.

```conda activate coda```

## <a name="GETTING_STARTED"></a>Getting Started

The [GETTING_STARTED](./docs/GETTING_STARTED.md) documentation describes how to download CODa programmatically
and use the visualization scripts.

## <a name="DATA_REPORT"></a>Data Report

The [DATA_REPORT](./docs/DATA_REPORT.md) documentation describes the contents and file structure of CODa. It
is important to read this before using the dataset.

## <a name="Models"></a>Models

To run the 3D object detection models from CODa, refer to CODa's sister Github repository: [coda-models](https://github.com/ut-amrl/coda-models). This repo provides the <b>benchmarks</b>, <b>pretrained_weights</b>, and 
<b>model training configurations</b> to reproduce the results in our paper.

# Citation
If you use our dataset of the tools, we would appreciate if you cite both our paper and dataset

### Paper Citation
```
@inproceedings{zhangcoda2023,
  author = {A. Zhang, C. Eranki, C. Zhang, R. Hong, P. Kalyani, L. Kalyanaraman, A. Gamare, A. Bagad, M. Esteva, J. Biswas},
  title = {{Towards Robust 3D Robot Perception in Urban Environments: The UT Campus Object Dataset}},
  booktitle = {},
  year = {2023}
}
``` 

### Dataset Citation
```
@data{T8/BBOQMV_2023,
author = {Zhang, Arthur and Eranki, Chaitanya and Zhang, Christina and Hong, Raymond and Kalyani, Pranav and Kalyanaraman, Lochana and Gamare, Arsh and Bagad, Arnav and Esteva, Maria and Biswas, Joydeep},
publisher = {Texas Data Repository},
title = {{UT Campus Object Dataset (CODa)}},
year = {2023},
version = {DRAFT VERSION},
doi = {10.18738/T8/BBOQMV},
url = {https://doi.org/10.18738/T8/BBOQMV}
}
```
