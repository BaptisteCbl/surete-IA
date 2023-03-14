from src.tf2.run import run_fashion_conf, run_fashion_noise

RUN_TYPE = "Confidence" #@param ["Confidence", "Noise"] {type:"raw"}
EPOCHS = 20 #@param {type:"slider", min:0, max:20, step:1}
BATCH_SIZE = 128 #@param ["512", "256", "128"] {type:"raw"}
ADVERSARIAL_TRAINING = False #@param {type:"boolean"}
EPSILON = 0.3 #@param {type:"slider", min:0, max:1, step:0.1}
LAMBDA = 2 #@param {type:"slider", min:0, max:3, step:0.1}
NOISY_MODEL = False #@param {type:"boolean"}
ADVERSARIAL_TRAINING_COEFFICIENT = 0.5 #@param {type:"slider", min:0, max:1, step:0.1}

if __name__ == "__main__":

    kwargs = dict(
        nb_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        adv_train=ADVERSARIAL_TRAINING,
        eps=EPSILON,
        lmbda=LAMBDA,
        noise=NOISY_MODEL,
        adv_train_coeff=ADVERSARIAL_TRAINING_COEFFICIENT,
    )

    if RUN_TYPE == "Confidence":
        run_fashion_conf(**kwargs)
    elif RUN_TYPE == "Noise":
        run_fashion_noise(**kwargs)