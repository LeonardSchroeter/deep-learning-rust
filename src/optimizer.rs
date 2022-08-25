use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&self, weights: Vec<&mut Tensor>);
}

pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&self, weights: Vec<&mut Tensor>) {
        for tensor in weights {
            *tensor -= &tensor.gradient().unwrap() * self.learning_rate;
        }
    }
}
