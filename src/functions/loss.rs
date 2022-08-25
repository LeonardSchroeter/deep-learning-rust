use ndarray::{ArrayD, Axis, IxDyn};

use crate::{function::Function, impl_tensor_binary, ndarray_util::ArcArrayD, tensor::Tensor};

#[derive(Default)]
struct MSE {
    y_hat: Option<ArcArrayD<f64>>,
    y: Option<ArcArrayD<f64>>,
}

impl Function for MSE {
    fn call(&mut self, input: Vec<&Tensor>) -> Tensor {
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

#[derive(Default)]
struct CrossEntropyWithSoftmax {
    y_hat: Option<ArcArrayD<f64>>,
    y: Option<ArcArrayD<f64>>,
}

impl Function for CrossEntropyWithSoftmax {
    fn call(&mut self, input: Vec<&Tensor>) -> Tensor {
        let y_hat = &input[0];
        let y = &input[1];

        self.y_hat = Some(y_hat.array.clone());
        self.y = Some(y.array.clone());

        let axis_to_sum = if y_hat.array.ndim() == 1 { 0 } else { 1 };

        let y_hat_exp = y_hat.array.mapv(|x| x.exp());
        let y_hat_exp_sum = y_hat_exp.sum_axis(Axis(axis_to_sum));

        let result = -&y.array
            * (&y_hat.array
                - y_hat_exp_sum
                    .insert_axis(Axis(axis_to_sum))
                    .mapv(|x| x.ln()));
        let result = result.sum_axis(Axis(axis_to_sum));

        Tensor::from_elem(&[], result.mean().unwrap())
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        let y_hat = self.y_hat.as_ref().unwrap();
        let y = self.y.as_ref().unwrap();

        let axis_to_sum = if y_hat.ndim() == 1 { 0 } else { 1 };

        let y_hat_exp = y_hat.mapv(|x| x.exp());
        let y_hat_exp_sum = y_hat_exp.sum_axis(Axis(axis_to_sum));

        let result = (y_hat_exp / y_hat_exp_sum.insert_axis(Axis(axis_to_sum)) - y)
            / y_hat.len_of(Axis(axis_to_sum)) as f64
            * &outer_gradient.array;

        vec![Tensor::new(result.clone()), Tensor::new(result * -1.0)]
    }

    fn arity(&self) -> u8 {
        2
    }
}

impl_tensor_binary!(
    MSE as mse,
    CrossEntropyWithSoftmax as cross_entropy_with_softmax
);

#[cfg(test)]
mod tests {
    use float_cmp::ApproxEq;

    use crate::tensor::Tensor;

    #[test]
    fn mse_call_works() {
        let zeros = Tensor::zeros(&[2, 2]);
        let ones = Tensor::ones(&[2, 2]);

        let mse = zeros.mse(&ones);

        let exptected = Tensor::from_elem(&[], 1.0);

        assert_eq!(mse, exptected);
    }

    #[test]
    fn mse_gradient_works() {
        let zeros = Tensor::zeros(&[2, 2]).require_grad();
        let ones = Tensor::ones(&[2, 2]).require_grad();

        let mse = zeros.mse(&ones);

        mse.backward();

        let grad_zeros = zeros.gradient().unwrap();
        let grad_ones = ones.gradient().unwrap();

        let exptected_zeros = Tensor::from_elem(&[2, 2], -0.5);
        let exptected_ones = Tensor::from_elem(&[2, 2], 0.5);

        assert_eq!(grad_zeros, exptected_zeros);
        assert_eq!(grad_ones, exptected_ones);
    }

    #[test]
    fn cross_entropy_call_works() {
        let input = Tensor::from_shape_vec(&[2, 2], vec![0.2, 0.8, 0.3, 0.7]);
        let target = Tensor::from_shape_vec(&[2, 2], vec![1.0, 0.0, 1.0, 0.0]);

        let cross_entropy = input.cross_entropy_with_softmax(&target);

        let expected = Tensor::from_elem(&[], 0.9753);

        assert!(cross_entropy.approx_eq(expected, (1e-4f64, 0)));
    }

    #[test]
    fn cross_entropy_gradient_works() {
        let input = Tensor::from_shape_vec(&[2, 2], vec![0.2, 0.8, 0.3, 0.7]).require_grad();
        let target = Tensor::from_shape_vec(&[2, 2], vec![1.0, 0.0, 1.0, 0.0]).require_grad();

        let cross_entropy = input.cross_entropy_with_softmax(&target);

        cross_entropy.backward();

        let grad_input = input.gradient().unwrap();

        let exptected_input =
            Tensor::from_shape_vec(&[2, 2], vec![-0.3228, 0.3228, -0.2993, 0.2993]);

        assert!(grad_input.approx_eq(exptected_input, (1e-4f64, 0)));
    }
}
