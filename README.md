# DGI
Deep Graph Infomax (Veličković *et al.*, ICLR 2019): [https://arxiv.org/abs/1809.10341](https://arxiv.org/abs/1809.10341)

![](http://www.cl.cam.ac.uk/~pv273/images/DGI.png)

## Overview
Here we provide an implementation of Deep Graph Infomax (DGI) in PyTorch, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:
- `data/` contains the necessary dataset files for Cora;
- `models/` contains the implementation of the DGI pipeline (`dgi.py`) and our logistic regressor (`logreg.py`);
- `layers/` contains the implementation of a GCN layer (`gcn.py`), the averaging readout (`readout.py`), and the bilinear discriminator (`discriminator.py`);
- `utils/` contains the necessary processing subroutines (`process.py`).

Finally, `execute.py` puts all of the above together and may be used to execute a full training run on Cora.

## Reference
If you make advantage of DGI in your research, please cite the following in your manuscript:

```
@inproceedings{
velickovic2018deep,
title="{Deep Graph Infomax}",
author={Petar Veli{\v{c}}kovi{\'{c}} and William Fedus and William L. Hamilton and Pietro Li{\`{o}} and Yoshua Bengio and R Devon Hjelm},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=rklz9iAcKQ},
}
```

## License
MIT
