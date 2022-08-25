use ndarray_rand::rand_distr::StandardNormal;

use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;

    fn trainable_weights(&mut self) -> Vec<&mut Tensor>;
}

pub struct Linear {
    weights: Tensor,
    bias: Tensor,
    activation: Option<fn(&Tensor) -> Tensor>,
}

impl Linear {
    pub fn new(
        input_dim: usize,
        layer_dim: usize,
        activation: Option<fn(&Tensor) -> Tensor>,
    ) -> Self {
        let weights = Tensor::random(&[input_dim, layer_dim], StandardNormal).require_grad();
        let bias = Tensor::random(&[layer_dim], StandardNormal).require_grad();

        Self {
            weights,
            bias,
            activation,
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let out = input.matmul(&self.weights).plus(&self.bias);

        if let Some(activation) = self.activation {
            activation(&out)
        } else {
            out
        }
    }

    fn trainable_weights(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.bias]
    }
}
