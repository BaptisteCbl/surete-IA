Code based on code from https://github.com/MadryLab/relu_stable
 **Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability**
_Kai Xiao, Vincent Tjeng, Nur Muhammad Shafiullah, Aleksander Madry_
https://arxiv.org/abs/1809.03008
International Conference on Learning Representations (ICLR), 2019

### Workflow

**Model Training**
1. `python train.py`: trains a model using parameters in `config.json`.

      Description of the defaults in `config.json`:
      
            1. It trains a 3-hidden layer convolutional architecture on MNIST or FashionMNIST.
            
            2. It uses adversarial training, L1 regularization, and ReLU stability regularization.
            
            3. The model is saved in `trained_models/$dataset/trained_model$i` with $datset being the dataset set in config.json and i the first natural number such that no trained model preexist.

2. `python post_process_model.py --model_dir $MODELDIR` : apply post-processing, converting the model from $MODELDIR to a .mat file and saving it within the given trained model folder as `mat/model.mat`.
   
      A typical call would look like: `python post_process_model.py --model_dir "trained_models\mnist\trained_model0" --do_eval`

      Command-line flags are available to choose post-processing options. Type `python post_process_model.py -h` to see all options.

## Citing this Code
```
@article{xiao2019training,
  title={Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability},
  author={Xiao, Kai and Tjeng, Vincent and Shafiullah, Nur Muhammad and Madry, Aleksander},
  journal={ICLR},
  year={2019}
}
```
