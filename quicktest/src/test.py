from absl import app, flags

import ast
from src.utils import *

FLAGS = flags.FLAGS


def main(_):

    x = FLAGS.FGSM_params
    print(x)
    for e in x:
        param = ast.literal_eval(e)
        print(type(param), param)


if __name__ == "__main__":
    flags.DEFINE_list("eps", [0], "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_string("data", "", "The dataset to load.")
    flags.DEFINE_string("save", "", "The path to save the model.")
    flags.DEFINE_list("attacks", [], "List of all attacks to perform")

    flags.DEFINE_list("FGSM_params", [], "Parameters for the FGSM attack")

    app.run(main)
