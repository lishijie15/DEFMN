## Net Load Forecasting in Large-scale Power Distribution Systems via a Dual-branch Experts Fusion Memory Network

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

To run our model, please execute the following code in the directory as `./DEFMN`.

```
python DEFMN_main.py 
```


