#  🌏 Cross-Domain Foundation Model Adaptation: Pioneering Computer Vision Models for Geophysical Data Analysis


🏢 [Computational Interpretation Group (CIG)](https://cig.ustc.edu.cn/main.htm) 

[Zhixiang Guo<sup>1</sup>](https://cig.ustc.edu.cn/guo/list.htm), 
[Xinming Wu<sup>1*</sup>](https://cig.ustc.edu.cn/xinming/list.htm), 
[Luming Liang<sup>2</sup>](https://www.microsoft.com/en-us/research/people/lulian/), 
[Hanlin Sheng<sup>1</sup>](https://cig.ustc.edu.cn/hanlin/list.htm), 
[Nuo Chen<sup>1</sup>](https://cig.ustc.edu.cn/nuo/list.htm), 
[Zhengfa Bi<sup>3</sup>](https://profiles.lbl.gov/416831-zhengfa-bi)

School of Earth and Space Sciences, University of Science and Technology of China, Hefei, China 
<img src="https://github.com/ProgrammerZXG/Cross-Domain-Foundation-Model-Adaptation/assets/89449763/399d6c3b-07eb-49dd-b0e9-d2bdb3cb3553" alt="中国科学技术大学_64x64" width="26" height="26">


Microsoft Applied Sciences Group, Redmond, WA 98052, United States
<img src="https://avatars.githubusercontent.com/u/6154722?s=200&v=4" width="26" height="26"> 

Lawrence Berkeley National Laboratory, 1 Cyclotron Rd, CA 94707, USA
<img width="30" alt="截屏2024-07-07 13 12 39" src="https://github.com/ProgrammerZXG/Cross-Domain-Foundation-Model-Adaptation/assets/89449763/2105a42f-7091-4910-819e-7e85b08f6639">

## :mega: News
:flying_saucer: The dataset, model, code, and demo are coming soon! 

:collision: [2024.07.07]: Github Repository Initialization. 

## :sparkles: Introduction
<p align="justify">
Workflow for adapting pre-trained foundation models to geophysics.
First, we prepare geophysical training datasets (1st column), 
which involves collecting and processing relevant geophysical data 
to ensure it is suitable for adaption fine-tuning. Next, we load the pre-trained 
foundation model as the data feature encoder (2nd column) 
and fine-tune the model to make it adaptable to geophysical data. 
To map the encoder features to the task-specific targets, 
we explore suitable decoders 
(3rd column) for geophysical downstream adaption. Finally, the adapted model 
is applied to various downstream tasks within the geophysics 
field (4th column).
</p>

<div align=center>
  <img src="https://github.com/ProgrammerZXG/Cross-Domain-Foundation-Model-Adaptation/assets/89449763/5d921c4c-c012-4cea-ad92-ae8b391ba78b" width="1000">
</div>


##  🚀 Quick Start

## :package: Dataset

## :bookmark: Citation

## :memo: Acknowledgment
This study is strongly supported by the Supercomputing 
Center of the University of Science and Technology of China, 
particularly with the provision of Nvidia 80G A100 GPUs, 
which are crucial for our experiments. 
We also thank [SEAM](https://seg.org/SEAM) for providing the seismic facies classification dataset, 
[TGS](https://www.kaggle.com/competitions/tgs-salt-identification-challenge) for the geobody identification dataset, 
[CAS](https://moon.bao.ac.cn) for the crater detection dataset, 
[Biondi](https://www.science.org/doi/full/10.1126/sciadv.adi9878) for the DAS seismic event detection dataset, 
and [CIG](https://cig.ustc.edu.cn/main.htm) for the deep fault detection dataset.

## :postbox: Contact
If you have any questions about this work, 
please feel free to contact xinmwu@ustc.edu.cn or zxg3@mail.ustc.edu.cn.
