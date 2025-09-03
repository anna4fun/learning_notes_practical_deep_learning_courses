# Key take-aways
1. configure the dimensions of each tensors objects carefully

2. GPU takes floats, so everything needs to be floats

3. check the scale of weights and gradients, if the scale differs too much, select a big learning_rate to step

4. In vision tasks, normalize the Pixels to [0,1], avoids saturation in nonlinearities,
keeps gradients at useful magnitudes,
and plays nicely with optimizers like SGD/Adam.
   * keeping original scale: would make the raw logits too big and push the sigmoid to 0 or 1. gradients could also vanish and training unstable. it could overflow (value > 65000 would become `Inf`)
   * scaling down too much: signals can vanish relative to float precision

5.last but not least, save all the outputs with a history of all the variations that I tried for more efficient experiments
