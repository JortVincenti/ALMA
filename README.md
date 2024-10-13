# [**DSnoT**](https://github.com/zyxxmu/DSnoT): Model Compression for Machine Translation in Large Language Models
Note: We build DSnoT on top of [**ALMA**](https://github.com/jortVincenti/ALMA/).
## Findings

In this work, we explored the application of various compression techniques to ALMA. Our study focused on five methods: GPTQ, Q-LoRA, SmoothQuant, Wanda, and DSnot. The results show that quantization techniques offer reductions in memory usage with minimal impact on translation quality. Pruning methods like Wanda provided faster inference times but at the cost of translation performance. The findings highlight the trade-offs between model efficiency and performance, demonstrating that compression methods can make large models like ALMA more accessible for deployment without sacrificing translation quality.


## Installation
### Clone the repo
```bash
git clone git@github.com:JortVincenti/ALMA.git
```
### Checkout to pruning branch
```bash
git checkout pruning
```
### Navigate to the DSnoT repository 
```bash
cd DSnoT
```
### Install Environment
Use the environment.yaml file

## Pruning with DSnoT
```bash
sbatch run_method.job
```
## Getting the Evaluation Result for the DSnoT-like pruned ALMA-7B
```bash
sbatch run_eval.job
```
## GPTQ quantization on ALMA 
[GPTQ Repository for ALMA Quantization](https://github.com/MatteoNulli/gptq)

## Getting the Evaluation Result for the GPTQ-like pruned ALMA-7B
```bash
cd scripts_gptq
bash launch_eval_pair-gptq.sh
```