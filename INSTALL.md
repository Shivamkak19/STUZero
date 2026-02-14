# Installation

## Prerequisites & Installation

```bash
conda create -n ezv2 python=3.8
conda activate ezv2
pip install -r requirements_a100.txt
```

Before starting training, you need to build the c++/cython style external packages. 
```
cd ez/mcts/ctree
bash make.sh
cd -
```

## Compilation Instructions

1. In addition to compiling `ctree`, also compile `ctree_v2` (If you find the training process is stuck, this issue exists in some cases.):
   ```bash
   cd ez/mcts/ctree_v2
   sh make.sh
   ```
1. In addition to compiling `ctree`, also compile `ori_ctree` :
   ```bash
   cd ez/mcts/ori_ctree
   sh make.sh
   ```
