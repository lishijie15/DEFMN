## Efficient Net Load Forecasting in Large-scale Power Distribution Systems via Dual-branch Experts Fusion Memory Network
## Abstract
> *Precise and efficient net load forecasting is crucial for Power Distribution Systems (PDS) to address the various challenges posed by increasing Renewable Energy Sources (RES) penetration. Existing studies on net load forecasting often overlook the characteristic differences (i.e., variable heterogeneity) between loads and Distributed Generation (DG), and the differences in graph structures (i.e., spatio-temporal heterogeneity) caused by changes in node net loads. These studies typically assume constant node properties and use the same model with shared parameters to extract spatio-temporal features for different variables, which limits the representation of the features. To address these challenges, this study proposes a novel model named Dual-branch Expert Fusion Memory Network (DEFMN). Specifically, this model designs customized experts (feature branch) with independent parameters for loads and different types of DG (variable branch) to extract features according to the characteristics of various RES. We also employ shared parameters modules, including series embedding, meta spatial memory, and decoder, to fully capture the spatio-temporal correlations between loads and DG, effectively learning their variable heterogeneity as well as spatio-temporal heterogeneity. Additionally, we propose a novel model named load-DG coupling model, which aims to construct a new large-scale PDS with different RES penetration based on the IEEE 8500-node test feeder. Extensive experiments are conducted on PDSs with varying penetration, and the proposed DEFMN consistently achieves state-of-the-art performance across various challenging scenarios.*
> 
![image](https://github.com/lishijie15/DEFMN/blob/ba745b7380de1ae4a9ee5819471696b75e07d402/pictures/DEFMN.png)
> 
### Introduction of Our Model

* To accurately predict net load in large-scale PDSs, we propose the DEFMN model, which employs an indirect strategy (i.e., each variable is predicted individually) to fully capture the spatio-temporal features of different variables and their interactions. 
* To effectively capture variable heterogeneity, we design dual-branch customized experts with independent parameters, employing different modules including Adaptive GCN (AGCN), temporal attention, and Temporal Channel Fusion (TCF) to extract spatio-temporal features of different variables based on their characteristics from both spatial and temporal dimensions.
* To fully explore the latent correlations between variables, we utilize series embedding with shared parameters to enhance the relational expression among different variables. Leveraging meta spatial memory and decoder with shared parameters tailored for spatio-temporal heterogeneity, we continuously update the dynamic graph structure to adapt to the time-varying real spatial structure of the PDS.
* To closely fit the real PDS, we propose LDCM, which constructs a new large-scale PDS with different RES penetration based on the IEEE 8500-node test feeder. The proposed DEFMN achieves State-of-The-Art (SOTA) performance across various PDSs.


### Installation and Run

#### Datasets

The complete load and DG datasets will be made publicly available later. The performance of the proposed model can be tested using partial univariate datasets (e.g., EXPY-TKY in MegaCRN). However, univariate datasets may not fully demonstrate the capabilities of DEFMN.

#### Requirements

DEFMN is compatible with PyTorch==1.13 versions.

All neural networks are implemented using PyTorch and trained on a single NVIDIA H100 80GB GPU.

Please code as below to install some nesessary libraries.

```
pip install -r requirements.txt
```



#### Run Our Model

To run our model, please execute the following code in the directory as `./`.

```
python DEFMN_main.py 
```
## BibTeX
If you find our work useful in your research. Please consider giving a star ⭐ and citation 📚.

```bash
@ARTICLE{11124332,
  author={Li, Shijie and Hu, Ruican and Chen, Guanlin and Chen, Lulu and Li, He and Jiang, Huaiguang and Xue, Ying and Kang, Jiawen and Zhang, Jun and Gao, David Wenzhong},
  journal={IEEE Transactions on Power Systems}, 
  title={Efficient Net Load Forecasting in Large-Scale Power Distribution Systems via Dual-Branch Experts Fusion Memory Network}, 
  year={2026},
  volume={41},
  number={1},
  pages={70-81},
  doi={10.1109/TPWRS.2025.3598366}
}
```

