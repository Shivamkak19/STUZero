====================
BASIC INSTALLATION
====================
conda create -n ezv2 python=3.8
conda activate ezv2

pip install -r req_upd.txt

cd ez/mcts/ctree
bash make.sh
cd -

cd ez/mcts/ctree_v2
sh make.sh

cd ez/mcts/ori_ctree
sh make.sh

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

wandb login
--> train.py offline mode wandb since della has no internet access
<!-- make changes the config file -->
python ez/train.py exp_config=ez/config/exp/atari.yaml 

