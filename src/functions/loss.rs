use ndarray::{ArrayD, IxDyn};

use crate::{function::Function, impl_tensor_binary, ndarray_util::ArcArrayD, tensor::Tensor};

#[derive(Default)]
struct MSE {
    y_hat: Option<ArcArrayD<f64>>,
    y: Option<ArcArrayD<f64>>,
}

impl Function for MSE {
    fn call(&mut self, input: Vec<&mut Tensor>) -> Tensor {
        let y_hat = &input[0];
        let y = &input[1];

        self.y_hat = Some(y_hat.array.clone());
        self.y = Some(y.array.clone());

        let diff = &y_hat.array - &y.array;
        let square = &diff * &diff;
        let result = square.sum() / square.len() as f64;

        Tensor::new(ArrayD::<f64>::from_elem(IxDyn(&[]), result))
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        let diff = self.y_hat.as_ref().unwrap() - self.y.as_ref().unwrap();
        let size = diff.len() as f64;
        let result = diff / size * 2.0;
        let result = result * &outer_gradient.array;

        vec![Tensor::new(result.clone()), Tensor::new(result * -1.0)]
    }

    fn arity(&self) -> u8 {
        2
    }
}

impl_tensor_binary!(MSE as mse);

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn mse_call_works() {
        let mut zeros = Tensor::zeros(&[2, 2]);
        let mut ones = Tensor::ones(&[2, 2]);

        let mse = zeros.mse(&mut ones);

        let exptected = Tensor::from_elem(&[], 1.0);

        assert_eq!(mse, exptected);
    }

    #[test]
    fn mse_gradient_works() {
        let mut zeros = Tensor::zeros(&[2, 2]);
        let mut ones = Tensor::ones(&[2, 2]);
        zeros.require_grad();
        ones.require_grad();

        let mse = zeros.mse(&mut ones);

        mse.backward();

        let grad_zeros = zeros.gradient().unwrap();
        let grad_ones = ones.gradient().unwrap();

        let exptected_zeros = Tensor::from_elem(&[2, 2], -0.5);
        let exptected_ones = Tensor::from_elem(&[2, 2], 0.5);

        assert_eq!(grad_zeros, exptected_zeros);
        assert_eq!(grad_ones, exptected_ones);
    }
}
