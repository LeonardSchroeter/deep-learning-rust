use ndarray::{ArrayD, Ix1, Ix2, IxDyn};

use crate::{
    function::Function,
    impl_tensor_binary, impl_tensor_unary,
    ndarray_util::{broadcast_backwards, ArcArrayD},
    tensor::Tensor,
};

#[derive(Default)]
pub struct Add {
    shape1: Vec<usize>,
    shape2: Vec<usize>,
}

impl Function for Add {
    fn call(&mut self, input: Vec<&Tensor>) -> Tensor {
        let input1 = &input[0];
        let input2 = &input[1];

        self.shape1 = input1.array.shape().to_vec();
        self.shape2 = input2.array.shape().to_vec();

        let result = &input1.array + &input2.array;

        Tensor::new(result)
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        vec![
            outer_gradient.broadcast_backwards(self.shape1.clone()),
            outer_gradient.broadcast_backwards(self.shape2.clone()),
        ]
    }

    fn arity(&self) -> u8 {
        2
    }
}

#[derive(Default)]
pub struct Times {
    input1: Option<ArcArrayD<f64>>,
    input2: Option<ArcArrayD<f64>>,
}

impl Function for Times {
    fn call(&mut self, input: Vec<&Tensor>) -> Tensor {
        let input1 = &input[0];
        let input2 = &input[1];

        self.input1 = Some(input1.array.clone());
        self.input2 = Some(input2.array.clone());

        Tensor::new(&input1.array * &input2.array)
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        vec![
            Tensor::new(broadcast_backwards(
                &(self.input2.as_ref().unwrap() * &outer_gradient.array),
                self.input1.as_ref().unwrap().shape().to_vec(),
            )),
            Tensor::new(broadcast_backwards(
                &(self.input1.as_ref().unwrap() * &outer_gradient.array),
                self.input2.as_ref().unwrap().shape().to_vec(),
            )),
        ]
    }

    fn arity(&self) -> u8 {
        2
    }
}

#[derive(Default)]
pub struct Sum {
    shape: Vec<usize>,
}

impl Function for Sum {
    fn call(&mut self, input: Vec<&Tensor>) -> Tensor {
        self.shape = input[0].array.shape().to_vec();

        Tensor::new(ArrayD::<f64>::from_elem(IxDyn(&[]), input[0].array.sum()))
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        vec![Tensor::new(
            ArrayD::<f64>::ones(IxDyn(&self.shape)) * &outer_gradient.array,
        )]
    }

    fn arity(&self) -> u8 {
        1
    }
}

#[derive(Default)]
pub struct MatMul {
    input1: Option<ArcArrayD<f64>>,
    input2: Option<ArcArrayD<f64>>,
}

impl Function for MatMul {
    fn call(&mut self, input: Vec<&Tensor>) -> Tensor {
        let input1 = &input[0];
        let input2 = &input[1];

        self.input1 = Some(input1.array.clone());
        self.input2 = Some(input2.array.clone());

        let ndim_lhs = input1.array.ndim();
        let ndim_rhs = input2.array.ndim();

        match (ndim_lhs, ndim_rhs) {
            (0, _) | (_, 0) => Tensor::new(&input1.array * &input2.array),
            (1, 1) => Tensor::from_elem(
                &[],
                input1
                    .array
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .ok()
                    .unwrap()
                    .dot(
                        &input2
                            .array
                            .clone()
                            .into_dimensionality::<Ix1>()
                            .ok()
                            .unwrap(),
                    ),
            ),
            (1, 2) => Tensor::new(
                input1
                    .array
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .ok()
                    .unwrap()
                    .dot(
                        &input2
                            .array
                            .clone()
                            .into_dimensionality::<Ix2>()
                            .ok()
                            .unwrap(),
                    )
                    .into_dyn(),
            ),
            (2, 1) => Tensor::new(
                input1
                    .array
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .ok()
                    .unwrap()
                    .dot(
                        &input2
                            .array
                            .clone()
                            .into_dimensionality::<Ix1>()
                            .ok()
                            .unwrap(),
                    )
                    .into_dyn(),
            ),
            (2, 2) => Tensor::new(
                input1
                    .array
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .ok()
                    .unwrap()
                    .dot(
                        &input2
                            .array
                            .clone()
                            .into_dimensionality::<Ix2>()
                            .ok()
                            .unwrap(),
                    )
                    .into_dyn(),
            ),
            (_, _) => todo!(),
        }
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        let ndim_lhs = self.input1.as_ref().unwrap().ndim();
        let ndim_rhs = self.input2.as_ref().unwrap().ndim();

        match (ndim_lhs, ndim_rhs) {
            (0, _) | (_, 0) => todo!(),
            (1, 1) => todo!(),
            (1, 2) => {
                let outer_gradient = outer_gradient
                    .array
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .unwrap();

                let input1 = self
                    .input1
                    .as_ref()
                    .unwrap()
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .unwrap();
                let input2 = self
                    .input2
                    .as_ref()
                    .unwrap()
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .unwrap();

                let input1_len = input1.len();
                let outer_gradient_len = outer_gradient.len();

                vec![
                    Tensor::new(outer_gradient.dot(&input2.reversed_axes()).into_dyn()),
                    Tensor::new(
                        input1
                            .into_shape((input1_len, 1))
                            .unwrap()
                            .dot(&outer_gradient.into_shape((1, outer_gradient_len)).unwrap())
                            .into_dyn(),
                    ),
                ]
            }
            (2, 1) => {
                let outer_gradient = outer_gradient
                    .array
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .unwrap();

                let input1 = self
                    .input1
                    .as_ref()
                    .unwrap()
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                let input2 = self
                    .input2
                    .as_ref()
                    .unwrap()
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .unwrap();

                let outer_gradient_len = outer_gradient.len();
                let input2_len = input2.len();

                vec![
                    Tensor::new(
                        outer_gradient
                            .clone()
                            .into_shape((outer_gradient_len, 1))
                            .unwrap()
                            .dot(&input2.into_shape((1, input2_len)).unwrap())
                            .into_dyn(),
                    ),
                    Tensor::new(input1.reversed_axes().dot(&outer_gradient).into_dyn()),
                ]
            }
            (2, 2) => {
                let outer_gradient = outer_gradient
                    .array
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                let input1 = self
                    .input1
                    .as_ref()
                    .unwrap()
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                let input2 = self
                    .input2
                    .as_ref()
                    .unwrap()
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .unwrap();

                vec![
                    Tensor::new(outer_gradient.dot(&input2.reversed_axes()).into_dyn()),
                    Tensor::new(input1.reversed_axes().dot(&outer_gradient).into_dyn()),
                ]
            }
            (_, _) => todo!(),
        }
    }

    fn arity(&self) -> u8 {
        2
    }
}

impl_tensor_unary!(Sum as sum);
impl_tensor_binary!(Add as plus, Times as times, MatMul as matmul);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_same_shape() {
        let a = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]);
        let b = Tensor::from_shape_vec(&[3], vec![4., 5., 6.]);
        let c = Tensor::from_shape_vec(&[3], vec![5., 7., 9.]);

        assert_eq!(a.plus(&b), c);
    }

    #[test]
    fn add_broadcast() {
        let a = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]);
        let b = Tensor::from_shape_vec(&[1], vec![4.]);
        let c = Tensor::from_shape_vec(&[3], vec![5., 6., 7.]);

        assert_eq!(a.plus(&b), c);
    }

    #[test]
    fn add_broadcast_2() {
        let a = Tensor::from_shape_vec(&[1], vec![1.]);
        let b = Tensor::from_shape_vec(&[3], vec![4., 5., 6.]);
        let c = Tensor::from_shape_vec(&[3], vec![5., 6., 7.]);

        assert_eq!(a.plus(&b), c);
    }

    #[test]
    fn add_gradient_same_shape() {
        let a = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).require_grad();
        let b = Tensor::from_shape_vec(&[3], vec![4., 5., 6.]).require_grad();

        let c = a.plus(&b).sum();

        c.backward();

        assert_eq!(
            a.gradient().unwrap(),
            Tensor::from_shape_vec(&[3], vec![1., 1., 1.])
        );

        assert_eq!(
            b.gradient().unwrap(),
            Tensor::from_shape_vec(&[3], vec![1., 1., 1.])
        );
    }

    #[test]
    fn add_gradient_broadcast() {
        let a = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).require_grad();
        let b = Tensor::from_shape_vec(&[], vec![4.]).require_grad();

        let c = a.plus(&b).sum();

        c.backward();

        assert_eq!(
            a.gradient().unwrap(),
            Tensor::from_shape_vec(&[3], vec![1., 1., 1.])
        );

        assert_eq!(b.gradient().unwrap(), Tensor::from_shape_vec(&[], vec![3.]));
    }

    #[test]
    fn mul_same_shape() {
        let a = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]);
        let b = Tensor::from_shape_vec(&[3], vec![4., 5., 6.]);
        let c = Tensor::from_shape_vec(&[3], vec![4., 10., 18.]);

        assert_eq!(a.times(&b), c);
    }

    #[test]
    fn mul_broadcast() {
        let a = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]);
        let b = Tensor::from_shape_vec(&[1], vec![4.]);
        let c = Tensor::from_shape_vec(&[3], vec![4., 8., 12.]);

        assert_eq!(a.times(&b), c);
    }

    #[test]
    fn mul_broadcast_2() {
        let a = Tensor::from_shape_vec(&[1], vec![1.]);
        let b = Tensor::from_shape_vec(&[3], vec![4., 5., 6.]);
        let c = Tensor::from_shape_vec(&[3], vec![4., 5., 6.]);

        assert_eq!(a.times(&b), c);
    }

    #[test]
    fn mul_gradient_same_shape() {
        let a = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).require_grad();
        let b = Tensor::from_shape_vec(&[3], vec![4., 5., 6.]).require_grad();

        let c = a.times(&b).sum();

        c.backward();

        assert_eq!(a.gradient().unwrap(), b);

        assert_eq!(b.gradient().unwrap(), a);
    }

    #[test]
    fn mul_gradient_broadcast() {
        let a = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).require_grad();
        let b = Tensor::from_shape_vec(&[], vec![4.]).require_grad();

        let c = a.times(&b).sum();

        c.backward();

        assert_eq!(
            a.gradient().unwrap(),
            Tensor::from_shape_vec(&[3], vec![4., 4., 4.])
        );

        assert_eq!(b.gradient().unwrap(), Tensor::from_shape_vec(&[], vec![6.]));
    }

    #[test]
    fn sum() {
        let a = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]);
        let c = Tensor::from_shape_vec(&[], vec![6.]);

        assert_eq!(a.sum(), c);
    }

    #[test]
    fn sum_gradient() {
        let a = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).require_grad();

        let c = a.sum();
        c.backward();
        assert_eq!(
            a.gradient().unwrap(),
            Tensor::from_shape_vec(&[3], vec![1., 1., 1.])
        );
    }
}
