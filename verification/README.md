# [CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)


## Installation and activation (extracted from the github page)


```bash
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
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

Here a large number of option can be configured. I will just explain the most important for us.
