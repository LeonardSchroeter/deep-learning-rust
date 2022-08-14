use ndarray::{ArrayD, Ix1, Ix2, IxDyn};

use crate::{
    function::Function, impl_tensor_binary, impl_tensor_unary, ndarray_util::ArcArrayD,
    tensor::Tensor,
};

#[derive(Default)]
pub struct Add {
    shape1: Vec<usize>,
    shape2: Vec<usize>,
}

impl Function for Add {
    fn call(&mut self, input: Vec<&mut Tensor>) -> Tensor {
        let input1 = &input[0];
        let input2 = &input[1];

        self.shape1 = input1.array.shape().to_vec();
        self.shape2 = input2.array.shape().to_vec();

        Tensor::new(&input1.array + &input2.array)
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        vec![
            Tensor::new(ArrayD::<f64>::ones(IxDyn(&self.shape1)) * &outer_gradient.array),
            Tensor::new(ArrayD::<f64>::ones(IxDyn(&self.shape2)) * &outer_gradient.array),
        ]
    }

    fn arity(&self) -> u8 {
        2
    }
}

#[derive(Default)]
pub struct Mul {
    input1: Option<ArcArrayD<f64>>,
    input2: Option<ArcArrayD<f64>>,
}

impl Function for Mul {
    fn call(&mut self, input: Vec<&mut Tensor>) -> Tensor {
        let input1 = &input[0];
        let input2 = &input[1];

        self.input1 = Some(input1.array.clone());
        self.input2 = Some(input2.array.clone());

        Tensor::new(&input1.array * &input2.array)
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        vec![
            Tensor::new(self.input2.as_ref().unwrap().to_owned() * &outer_gradient.array),
            Tensor::new(self.input1.as_ref().unwrap().to_owned() * &outer_gradient.array),
        ]
    }

    fn arity(&self) -> u8 {
        2
    }
}

#[derive(Default)]
pub struct Square {
    input: Option<ArcArrayD<f64>>,
}

impl Function for Square {
    fn call(&mut self, input: Vec<&mut Tensor>) -> Tensor {
        let unary_input = &input[0];

        self.input = Some(unary_input.array.clone());

        Tensor::new(&unary_input.array * &unary_input.array)
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        vec![Tensor::new(
            self.input.as_ref().unwrap() * 2.0 * &outer_gradient.array,
        )]
    }

    fn arity(&self) -> u8 {
        1
    }
}

#[derive(Default)]
pub struct Sum {
    shape: Vec<usize>,
}

impl Function for Sum {
    fn call(&mut self, input: Vec<&mut Tensor>) -> Tensor {
        self.shape = input[0].array.shape().to_vec();

        Tensor::new(ArrayD::<f64>::from_elem(IxDyn(&[]), input[0].array.sum()))
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        vec![Tensor::new(ArrayD::<f64>::ones(IxDyn(&self.shape))).dot(outer_gradient)]
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
    fn call(&mut self, input: Vec<&mut Tensor>) -> Tensor {
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

impl_tensor_unary!(Square as square, Sum as sum);
impl_tensor_binary!(Add as add, Mul as mul, MatMul as matmul);
