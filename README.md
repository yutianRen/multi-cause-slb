# Self-Labeling in Multivariate Causality and Quantification for Adaptive Machine Learning

This repo includes the source codes and instructions of how to run the
simulation and experiments for the [paper](https://arxiv.org/abs/2404.05809).

**- 04/09/2024** Initial release, more instructions may be added in the future.

## To run the experiments

First, we need to download data from this [link](https://drive.google.com/drive/folders/16_TGalwFJDZDWyc-SkBY3i3aCJtYNJyf?usp=drive_link).

Also, a conda environment is suggested to be created by running:
```
conda create -n slb python=3.10
```
Then install the pacakges by
```
pip install -r requirements.txt
```
Note that the paper results are derived using torch==1.12.0+cu116. This repo has been updated to support torch>=2.0 
as well but not fully tested.

We can run the self-labeling by
```
python3 slb.py --pretrain_path <unperturbed dataset path>.pkl 
        --data_path <unperturbed dataset path>.pkl 
        --out_dir_path <output path>
        --rand_seed 0
        --add_data <addtional dataset path>.pkl 
```

For example, to use the wind-0.5 dataset as the perturbed set and the nowind as the unperturbed set, we can run
```
python3 slb.py --pretrain_path data/nowind_4500_lb4itm.pkl 
        --data_path data/wind-0.5_4500_lb4itm.pkl  
        --out_dir_path results/
        --rand_seed 0
        --add_data data/wind-0.5_7200_lb4itm.pkl
```

## Citation
Please consider citing this work if you find it helpful:
```
@misc{ren2024selflabeling,
      title={Self-Labeling in Multivariate Causality and Quantification for Adaptive Machine Learning}, 
      author={Yutian Ren and Aaron Haohua Yen and G. P. Li},
      year={2024},
      eprint={2404.05809},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
