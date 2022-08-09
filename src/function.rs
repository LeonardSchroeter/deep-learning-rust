use std::{cell::RefCell, fmt::Debug, rc::Rc};

use ndarray::{ArrayD, IxDyn};

use crate::{ndarray_util::ArcArrayD, node::Node, tensor::Tensor};

pub trait Function {
    fn arity(&self) -> u8;

    fn call(&mut self, inputs: Vec<&mut Tensor>) -> Tensor;

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor>;

    fn call_checked(&mut self, inputs: Vec<&mut Tensor>) -> Tensor {
        if inputs.len() != self.arity().into() {
            panic!("Wrong number of input tensors");
        }

        self.call(inputs)
    }

    fn call_and_build_graph_unary(input: &mut Tensor) -> Tensor
    where
        Self: Sized + Default + 'static,
    {
        let mut function = Self::default();

        let mut result = function.call_checked(vec![input]);

        if !input.requires_grad {
            return result;
        }

        result.requires_grad = true;

        let mut node = Node::new(Some(Box::new(function)));
        node.change_children(vec![(Some(Rc::clone(&input.node.as_ref().unwrap())))]);
        result.node = Some(Rc::new(RefCell::new(node)));

        result
    }

    fn call_and_build_graph_binary(input1: &mut Tensor, input2: &mut Tensor) -> Tensor
    where
        Self: Sized + Default + 'static,
    {
        let mut function = Self::default();

        let mut result = function.call_checked(vec![input1, input2]);

        if !input1.requires_grad && !input2.requires_grad {
            return result;
        }

        result.requires_grad = true;

        let mut node = Node::new(Some(Box::new(function)));

        let mut new_children = Vec::new();

        new_children.push(if input1.requires_grad {
            Some(Rc::clone(&input1.node.as_ref().unwrap()))
        } else {
            None
        });
        new_children.push(if input2.requires_grad {
            Some(Rc::clone(&input2.node.as_ref().unwrap()))
        } else {
            None
        });

        node.change_children(new_children);

        result.node = Some(Rc::new(RefCell::new(node)));

        result
    }
}

impl Debug for dyn Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dyn Function")
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

        input1.dot(input2)
    }

    fn gradient(&self, outer_gradient: &Tensor) -> Vec<Tensor> {
        vec![
            outer_gradient.dot(&Tensor::new(
                self.input2.as_ref().unwrap().to_owned().reversed_axes(),
            )),
            Tensor::new(self.input1.as_ref().unwrap().to_owned().reversed_axes())
                .dot(&outer_gradient),
        ]
    }

    fn arity(&self) -> u8 {
        2
    }
}

impl Tensor {
    pub fn square(&mut self) -> Tensor {
        Square::call_and_build_graph_unary(self)
    }

    pub fn add(&mut self, rhs: &mut Tensor) -> Tensor {
        Add::call_and_build_graph_binary(self, rhs)
    }

    pub fn mul(&mut self, rhs: &mut Tensor) -> Tensor {
        Mul::call_and_build_graph_binary(self, rhs)
    }

    pub fn sum(&mut self) -> Tensor {
        Sum::call_and_build_graph_unary(self)
    }

    pub fn matmul(&mut self, rhs: &mut Tensor) -> Tensor {
        MatMul::call_and_build_graph_binary(self, rhs)
    }
}
