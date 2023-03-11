# [α-β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)

![α-β-CROWN verifier](https://camo.githubusercontent.com/249374ab11a00eeb7d631b3c7cc195d52a75c39195d2c90db549ac506cf68942/68747470733a2f2f7777772e6875616e2d7a68616e672e636f6d2f696d616765732f75706c6f61642f616c7068612d626574612d63726f776e2f616263726f776e5f75736167652e706e67)

The verifier can use specificication sush as L norm, input bound and output linear constrainst (via VNNLIB file) 

## Installation and activation (extracted from the github page)


```bash
# git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
# conda init bash #surely useless on your system
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
# install all dependents into the alpha-beta-crown environment
conda env create -f complete_verifier/environment.yml --name alpha-beta-crown
# activate the environment
conda activate alpha-beta-crown
# run 
```

## Run a verification
```bash
cd complete_verifier
python abcrown.py --config exp_configs/"config".yaml
```

## Config file

### General

A large number of option can be configured. I will just explain the most important features for us.
The list of all options can be find [here](https://github.com/Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/abcrown_all_params.yaml).

### Basic examples

Some short config can be found in [this folder](https://github.com/Verified-Intelligence/alpha-beta-CROWN/tree/main/complete_verifier/exp_configs/tutorial_examples).

### Custom configuration

- gerenal (device,...)
- model
- data
- specification
- solver
- bab
- attack


Here is the config file for running the verifier on the FashionMNIST dataset with a custom save and a custom pytorch model.
```yaml
general:
  device: cpu
model:
  name: cnn_small  # This model is defined in model_defs.py. Add your own model definitions there.
  path: models/custom_pt/FashionMNIST_cnn_small_0.0392_pgd.pt  # Path to PyTorch checkpoint.
data:
  dataset: FashionMNIST  # Dataset name. This is just the standard CIFAR-10 test set defined in the "load_verification_dataset()" function in utils.py
    #mean: [0.4914, 0.4822, 0.4465]  # Mean for normalization.
    #std: [0.2471, 0.2435, 0.2616]  # Std for normalization.
  start: 0  # First example to verify in dataset.
  end: 100  # Last example to verify in dataset. We verify 100 examples in this test.
specification:
  norm: .inf  # Linf norm (can also be 2 or 1).
  epsilon: 0.0392 #0.00784313725490196  # epsilon=2./255.
    #attack:  # Currently attack is only implemented for Linf norm.
    #pgd_steps: 100  # Increase for a stronger attack. A PGD attack will be used before verification to filter on non-robust data examples.
    #pgd_restarts: 30  # Increase for a stronger attack.
solver:
  batch_size: 2048  # Number of subdomains to compute in parallel in bound solver. Decrease if you run out of memory.
  alpha-crown:
    iteration: 100   # Number of iterations for alpha-CROWN optimization. Alpha-CROWN is used to compute all intermediate layer bounds before branch and bound starts.
    lr_alpha: 0.1    # Learning rate for alpha in alpha-CROWN. The default (0.1) is typically ok.
  beta-crown:
    lr_alpha: 0.01  # Learning rate for optimizing the alpha parameters, the default (0.01) is typically ok, but you can try to tune this parameter to get better lower bound.
    lr_beta: 0.05  # Learning rate for optimizing the beta parameters, the default (0.05) is typically ok, but you can try to tune this parameter to get better lower bound.
    iteration: 20  # Number of iterations for beta-CROWN optimization. 20 is often sufficient, 50 or 100 can also be used.
bab:
  timeout: 120  # Timeout threshold for branch and bound. Increase for verifying more points.
  branching:  # Parameters for branching heuristics.
    reduceop: min  # Reduction function for the branching heuristic scores, min or max. Using max can be better on some models.
    method: kfsb  # babsr is fast but less accurate; fsb is slow but most accurate; kfsb is usually a balance.
    candidates: 3  # Number of candidates to consider in fsb and kfsb. More leads to slower but better branching. 3 is typically good enough.
```

## Custom models

The model named in the config file must be in ```complete_verifier/model_defs.py``` for pytorch models (not necessary for onnx files).

## Custom saves

To use ```pt``` saves from the adversial training, they must placed in the ```complete_verifier/models/custom_pt``` folder.
Then run the ```complete_verifier/models/custom_pt/change_state_dict.py``` script on them.

## Run the instances we used :

```bash
python abcrown.py --config exp_configs/tutorial_examples/basic_mnist_cpu.yaml
python abcrown.py --config exp_configs/tutorial_examples/basic_cifar_free_cpu.yaml        
python abcrown.py --config exp_configs/tutorial_examples/basic_cifar_none_cpu.yaml        
python abcrown.py --config exp_configs/tutorial_examples/basic_cifar_pgd_cpu.yaml         
python abcrown.py --config exp_configs/tutorial_examples/basic_fashion_mnist_pgd_cpu.yaml 
python abcrown.py --config exp_configs/tutorial_examples/basic_fashion_mnist_none_cpu.yaml
```

Please find the results in the ```res_*_*.md``` files.

# [treeVerification](https://github.com/chenhongge/treeVerification)


## Run a verification (extracted from the github page)

```bash
# git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd treeVerification
./treeVerify "config".json
```

## Config file

### General

A large number of option can be configured. I will just explain the most important features for us.
The list of all options can be find [here](https://github.com/Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/abcrown_all_params.yaml).

### Basic examples

Some short config can be found in the README.md.
### Custom configuration

Here is the config file for running the verifier on the FashionMNIST dataset with a robust trained tree:
```json
{   
    "inputs":       "./tree_verification_models/fashion_robust_new/0200.libsvm", 
    "model":        "./tree_verification_models/fashion_robust_new/0200.json",
    "start_idx":    990,
    "num_attack":   500,
    "eps_init":     0.0392156862745098,
    "max_clique":   2,
    "max_search":   10,
    "max_level":    1,
    "num_classes":  10
}
```
"inputs" refers to usually a test set in libsvm format and model the decision tree.
The test sets are extracted from the adversial training repository from the sames authors [RobustTree](https://github.com/chenhongge/RobustTrees).


## Run the instances we used :

```bash
./treeVerify config_mnist_rob.json
./treeVerify config_mnist_unrob.json
./treeVerify config_fashion_rob.json
./treeVerify config_fashion_unrob.json
```

Please find the results in the ```res.md``` file.
