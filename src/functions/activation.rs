use crate::{function::Function, impl_tensor_unary, ndarray_util::ArcArrayD, tensor::Tensor};

#[derive(Default)]
pub struct ReLU {
    input: Option<ArcArrayD<f64>>,
}

impl Function for ReLU {
    fn call(&mut self, input: Vec<&mut Tensor>) -> Tensor {
        let input = &input[0];

        self.input = Some(input.array.clone());

        Tensor::new(input.array.map(|x| if *x > 0.0 { *x } else { 0.0 }))
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        vec![Tensor::new(
            self.input
                .as_ref()
                .unwrap()
                .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
                * &outer_gradient.array,
        )]
    }

    fn arity(&self) -> u8 {
        1
    }
}

impl_tensor_unary!(ReLU as relu);

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn relu_call_works() {
        let mut zeros = Tensor::zeros(&[2, 3]);
        let relu_zeros = zeros.relu();

        let mut ones = Tensor::ones(&[2, 3]);
        let relu_ones = ones.relu();

        let mut tensor = Tensor::from_shape_vec(&[1, 2, 2], vec![-1.2, 0.0, 3.2, -2.0]);
        let relu_tensor = tensor.relu();
        let expected = Tensor::from_shape_vec(&[1, 2, 2], vec![0.0, 0.0, 3.2, 0.0]);

        assert_eq!(zeros, relu_zeros);
        assert_eq!(ones, relu_ones);
        assert_eq!(relu_tensor, expected);
    }

    #[test]
    fn relu_gradient_works() {
        let mut tensor = Tensor::from_shape_vec(&[1, 2, 2], vec![-1.2, 0.0, 3.2, -2.0]);
        tensor.require_grad();
        let mut relu_tensor = tensor.relu();
        relu_tensor.sum().backward();
        let gradient = tensor.gradient().unwrap();

        let expected = Tensor::from_shape_vec(&[1, 2, 2], vec![0.0, 0.0, 1.0, 0.0]);

        assert_eq!(gradient, expected);
    }
}
