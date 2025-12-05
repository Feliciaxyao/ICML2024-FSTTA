# ICML2024-FSTTA
Fast-Slow Test-time Adaptation for Online Vision-and-Language Navigation
## Introduction
![image](img/FSTTA.png)

### Fast-Slow Test-time Adaptation for Online Vision-and-Language Navigation

Junyu Gao, Xuan Yao, Changsheng Xu

State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences.

[Paper Link on ICML 2024](https://icml.cc/virtual/2024/poster/33723) 

## Usage

### 🔄 Update Notes for Detailed Modules of FSTTA

We have added additional comments and clarifying notes to the FAST and SLOW components of the FSTTA framework.
These notes help readers more easily align the code with the equations presented in the paper and understand how the adaptation pipeline is organized in practice.

The update includes:
- clearer explanations for gradient decomposition–accumulation,
- notes for dynamic learning-rate scaling,
- clarification on how parameter trajectories are maintained across samples,
- guidance on the construction of the reference direction for the SLOW update.

The underlying implementation logic remains unchanged, and all existing checkpoints, pretrained models, and scripts continue to work as before.

**(1) Fast Update Module**
- Gradient decomposition and concordant direction explanation: [FSTTA.py](https://github.com/Feliciaxyao/ICML2024-FSTTA/blob/a6aa554f6de2acd72af83d555cd3f94c05140397/map_nav_src/models/FSTTA.py#L76-L138)  
- Dynamic learning-rate scaling: [FSTTA.py](https://github.com/Feliciaxyao/ICML2024-FSTTA/blob/a6aa554f6de2acd72af83d555cd3f94c05140397/map_nav_src/models/FSTTA.py#L140-L168)  
- Usage details in rollout: [agent_obj.py](https://github.com/Feliciaxyao/ICML2024-FSTTA/blob/a6aa554f6de2acd72af83d555cd3f94c05140397/map_nav_src/reverie/agent_obj.py#L374-L382)

**(2) Slow Update Module**
- Parameter trajectory accumulation: [agent_obj.py](https://github.com/Feliciaxyao/ICML2024-FSTTA/blob/a6aa554f6de2acd72af83d555cd3f94c05140397/map_nav_src/reverie/agent_obj.py#L593-L597)
- Reference direction construction: [agent_obj.py](https://github.com/Feliciaxyao/ICML2024-FSTTA/blob/a6aa554f6de2acd72af83d555cd3f94c05140397/map_nav_src/reverie/agent_obj.py#L599-L607)  
- Slow principal update path: [FSTTA.py](https://github.com/Feliciaxyao/ICML2024-FSTTA/blob/a6aa554f6de2acd72af83d555cd3f94c05140397/map_nav_src/models/FSTTA.py#L308-L332)


### Prerequisites

1. Install Matterport3D simulators: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator). We use the latest version the same as DUET.
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

2. Install requirements:
```setup
conda create --name fsvln python=3.8.5
conda activate fsvln
```
* Required packages are listed in `requirements.txt`. You can install by running:

```
pip install -r requirements.txt
```

3. Please download data from [Dropbox](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0), including processed annotations, features and pretrained models of REVERIE datasets and R2R datasets. 
Before running the code, please put the data in `datasets' directory.

4. Please download pretrained LXMERT model by running:
```
mkdir -p datasets/pretrained 
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P datasets/pretrained
```


### Pretraining (Base Model)

Combine behavior cloning and auxiliary proxy tasks in pretraining:
```pretrain
cd pretrain_src
bash run_reverie.sh 
```



### Fine-tuning (Base Model)

Use pseudo interative demonstrator to fine-tune the model:
```finetune
cd map_nav_src
bash scripts/run_reverie.sh 
```

### Test-time Adaptation & Evaluation

Use pseudo interative demonstrator to equip the model with our FSTTA:
```TTA during test time
cd map_nav_src
bash scrips/run_reverie_tta.sh
```


## Acknowledgements
Our implementations are partially based on [VLN-DUET](https://github.com/cshizhe/VLN-DUET), [HM3DAutoVLN](https://github.com/cshizhe/HM3DAutoVLN) and [VLN-BEVBert](https://github.com/MarSaKi/VLN-BEVBert.git). Thanks to the authors for sharing their code.


## Related Work
* [Reverie: Remote embodied visual referring expression in real indoor environments](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qi_REVERIE_Remote_Embodied_Visual_Referring_Expression_in_Real_Indoor_Environments_CVPR_2020_paper.pdf)
* [Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments](https://arxiv.org/pdf/2004.02857)

## Citation

If you find this project useful in your research, please consider cite:
```
@inproceedings{Gao2024Fast,
  title={Fast-Slow Test-time Adaptation for Online Vision-and-Language Navigation},
  author={Junyu Gao and Xuan Yao and Changsheng Xu},
  journal={Proceedings of the 41st International Conference on Machine Learning},
  year={2024},
  url={}
}
```


