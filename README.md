# A deep learning library in Rust using ndarray as a backend

### Overview

This library provides an interface to easily implement deep learning algorithms aswell as implementations for some widely used algorithms. At the heart of this library lays the `Tensor` type which is a wrapper around ndarray's `Array` type with dynamic dimensions. Similar to PyTorch's autograd, performing operations on a tensor builds a directed acyclic computational graph. Calling `Tensor::backward` on a scalar tensor makes the DAG perform backpropagation to compute the gradients for each tensor on which the result depends.

### Basic example

The following is a basic example of constructing a basic feed forward network with three layers for learning the MNIST dataset.

```rust
// A struct which represents our feed forward network holding three linear layers.
pub struct MyNeuralNet {
    input_layer: Linear,
    hidden_layer: Linear,
    output_layer: Linear,
}

impl MyNeuralNet {
    fn new() -> Self {
        Self {
            input_layer: Linear::new(784, 128, Some(Tensor::sigmoid)),
            hidden_layer: Linear::new(128, 64, Some(Tensor::sigmoid)),
            output_layer: Linear::new(64, 10, None),
        }
    }
}

impl Layer for MyNeuralNet {
    // Each layer needs to implement the `forward` functions which simply implements the layers logic. In our case, the forward function calls the forward function of our three linear layers sequentially.
    fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.input_layer.forward(input);
        let x = self.hidden_layer.forward(&x);
        self.output_layer.forward(&x)
    }

    // A layer also needs to implement the `trainable_weights` function, which simply returns a vec of all of its trainable weights. In our case, it returns the trainable weights of our three linear layers. As you can see, we can easily define layers in terms of other layers which allows us to easily abstract away logic.
    fn trainable_weights(&mut self) -> Vec<&mut Tensor> {
        self.input_layer
            .trainable_weights()
            .into_iter()
            .chain(self.hidden_layer.trainable_weights().into_iter())
            .chain(self.output_layer.trainable_weights().into_iter())
            .collect()
    }
}

fn main() {
    // Import the MNIST dataset
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Build tensors with batch size 100 from data
    // Inputs are rescaled to values in [0, 1]
    let x = Tensor::from_shape_vec(
        &[500, 100, 784],
        trn_img.into_iter().map(|x| x as f64 / 256.).collect(),
    );

    let y = Tensor::from_shape_vec(
        &[500, 100, 10],
        trn_lbl.into_iter().map(|y| y as f64).collect(),
    );

    const EPOCHS: usize = 10;

    // Our neural network
    let mut nn = MyNeuralNet::new();

    // A simple stochastic gradient descent optimizer with a learning rate of 0.01
    let optimizer = SGD::new(0.01);

    for epoch in 1..EPOCHS {
        for (i, (x, y)) in x.array.outer_iter().zip(y.array.outer_iter()).enumerate() {
            let x = Tensor::new(x.to_owned());
            let y = Tensor::new(y.to_owned());

            // pass our batched input through the neural network
            let y_pred = nn.forward(&x);

            // compute the cross entropy loss
            let loss = y_pred.cross_entropy_with_softmax(&y);

            // perform backpropagation to compute the gradients
            loss.backward();

            // pass the trainable weights to out optimizer, the weights now hold the gradients which the optimizer will apply to the corresponding weights
            optimizer.step(nn.trainable_weights())
        }
    }
}
```
