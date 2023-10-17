# UT Campus Object Dataset (CODa)

<b>Official dataset development kit for CODa. We strongly recommend using this repository to visualize the
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
and use the visualization scripts. To download the tiny and small splits of CODa by individual files, go to the
[Texas Data Repository](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi%3A10.18738%2FT8%2FBBOQMV&version=DRAFT).

## <a name="DATA_REPORT"></a>Data Report

The [DATA_REPORT](./docs/DATA_REPORT.md) documentation describes the contents and file structure of CODa. It
is important to read this before using the dataset.

## <a name="Models"></a>Models

To run the 3D object detection models from CODa, refer to CODa's sister Github repository: [coda-models](https://github.com/ut-amrl/coda-models). This repo provides the <b>benchmarks</b>, <b>pretrained_weights</b>, and 
<b>model training configurations</b> to reproduce the results in our paper.

# Citation
If you use our dataset of the tools, we would appreciate if you cite both our [paper](https://arxiv.org/abs/2309.13549) and [dataset](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/BBOQMV).

### Paper Citation
```
@misc{zhang2023robust,
      title={Towards Robust Robot 3D Perception in Urban Environments: The UT Campus Object Dataset}, 
      author={Arthur Zhang and Chaitanya Eranki and Christina Zhang and Ji-Hwan Park and Raymond Hong and Pranav Kalyani and Lochana Kalyanaraman and Arsh Gamare and Arnav Bagad and Maria Esteva and Joydeep Biswas},
      year={2023},
      eprint={2309.13549},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
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

## Dataset Metadata
The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">UT CODa: UT Campus Object Dataset</code></td>
  </tr>
  <tr>
    <td>alternateName</td>
    <td><code itemprop="alternateName">UT Campus Object Dataset</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/ut-amrl/coda-devkit</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://amrl.cs.utexas.edu/coda/</code></td>
  </tr>
    <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/BBOQMV</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">The UT Campus Object Dataset is a large-scale multiclass, multimodal egocentric urban robot dataset operated by human operators under a variety of weather, lighting, and viewpoint variations. We release this dataset publicly and pretrained models to help advance egocentric perception and navigation research in urban environments.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Automous Mobile Robotics Laboratory</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://amrl.cs.utexas.edu</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">UT Campus Object Dataset License Agreement for Non-Commercial Use (Oct 2023)</code></td>
          </tr>
          <tr>
            <td>url</td>
            <td><code itemprop="url">https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/BBOQMV&version=1.2&selectTab=termsTab</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
</table>
</div>
