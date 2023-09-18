# Hierarchical Deep Counterfactual Regret Minimization

## Required environments:
- on Ubuntu 20.04
- Python 3.8
- requests
- torch 1.7.1
- gym 0.10.9
- pycrayon 0.5
- ray 1.8.0
- redis
- boto3
- psutil
- protobuf 3.20.0
- tqdm
- scipy
- tensorboard 2.9.0
- ...

## Comparisons with SOTA model-free baselines for zero-sum games
- To reproduce the comparison result on task 'XXX', you need to first enter the corresponding folder 'Main_XXX', where 'XXX' can be one of {'Leduc', 'FHP'}.
- For each task, it has subtypes varying in the game size or decision horizon. For 'Leduc', it has {'StandardLeduc', 'LargeLeduc_10', 'LargeLeduc_15', 'LargeLeduc_20'}. For 'FHP', it has {'Flop3Holdem', 'LargeFlop3Holdem'}. Note that 'LargeFlop3Holdem' corresponds to 'FHP_10' in the paper.
- To run the experiment with specific algorithms ('XXX' can be one of {'Leduc', 'FHP'}):

```bash
# HDCFR:
python XXX_HDCFR.py 
# DREAM:
python XXX_DREAM.py
# OSSDCFR:
python XXX_OSSDCFR.py
# NFSP
python XXX_NFSP.py
```

- To run experiments with a certain subtype 'YYY', please change the variables 'game_cls' and 'name_str' in the python scripts mentioned above accordingly.
- The plots will be saved as tensorboard files in the folder 'saved_data'.
- To get the Head2Head results, you need first to collect the training results as checkpoints and then utilize 'Leduc_H2H.py' or 'FHP_H2H.py' in each folder. 

## Ablation study results

- For the ablated version 'NO_MHA', please enter the folder 'AS_no_attn'.
- For the ablated version 'NO_BASELINE', please enter the folder 'AS_no_baseline'.
- For the ablated version 'CFR_RULE', please enter the folder 'AS_CFR_rule'.
- For other ablated versions, please enter the folder 'AS'.
- To run the experiment within the first three folders:
```bash
# HDCFR
python Leduc_HDCFR.py
```
- To run experiments with certain \epsilon ('XXX' can be one of {'0_00', '0_25', '0_50', '0_75', '1_00'}):
```bash
# HDCFR
python Leduc_HDCFR_XXX.py
```
- To run experiments with certain traj_num ('XXX' can be one of {'500', '1000', '2000', '3000', '4000'}):
```bash
# HDCFR
python Leduc_HDCFR_XXX.py
```
- The plots will be saved as tensorboard files in the folder 'saved_data'.

## Skill transfer results

- To generate pretrained model, please enter the folder 'Pretrained_model'. You can specify the subtype by modifying the variables 'game_cls' and 'name_str' in 'Leduc_HDCFR.py' accordingly.
- Or, you can directly use the pretrained models provided by us, which are available in 'Transfer_performance/saved_data'.
- To generate transfer learning results, please enter the folder 'Transfer_performance'. You can select a pretrained model by (un)commenting corresponding lines from Line 20 to Line 34 in 'Leduc_HDCFR.py'. You can also specify the 'Fixed' (Line 36-38) or 'Non-fixed' (Line 36-38) mode in that file. Then, you can run:
```bash
# HDCFR
python Leduc_HDCFR.py
```
- To evaluate the changing frequency of skills, please enter the folder 'Skill_change_freq'. You can specify the checkpoint you want to test in Line 20-34 of 'Leduc_HDCFR.py'. Then, you can run:
```bash
# HDCFR
python Leduc_HDCFR.py
```
- The plots will be saved as tensorboard files in the folder 'saved_data', while the numerical results will be shown in the terminal.

