## TODO List
- [ ] Change the ExtractFrom Transform to move information instead of copy
- [ ] Save logging in particular folder
- [ ] Add Communication Environments (lbforaging, pp, pcp, TJ)
- [ ] Extend GNN Module by overconfiguration, n_layers, Sequential Structure, node attributs...
- [ ] Implement DICG, MAGIC and CommFormer (without GNN method') with changing Experiments loop by Encoder -> Communication <-> Decoder -> Algo RL
- [ ] Add Optuna File to tune a range of hyper-parameters, and change the Benchmark to add the option of Tunning before running
- [ ] Rename BenchMARL as BenchMARL-Comm
- [ ] Additionnal Usefull logging for seing Attention Comm, Dynamic CG, reward by Agents

## Install

1. Need of python 3.11
```sh
conda env create -n p311 python=3.11.14 swig
conda activate p311
```
2. Install of torch 2.8
```sh
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```
3. Install of pyG
```sh
pip install torch_geometric

# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```
4. Install of MARL stuff
```sh
pip install torchrl benchmarl wandb moviepy
```
5. Install envs
```sh
pip install vmas dm-meltingpot lbforaging
pip install pettingzoo[all]==1.24.3
```
6. [Install SMACv2](https://github.com/facebookresearch/BenchMARL/blob/main/.github/unittest/install_smacv2.sh)

7. Install additional stuff
```sh
pip install matplotlib seaborn
```
8. Install marl-eval for plotting
```sh
pip install "git+https://github.com/instadeepai/marl-eval.git"
pip install numpy==2.2.6
```