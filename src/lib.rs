#![allow(dead_code)]

use core::panic;
use std::{cell::RefCell, fmt::Debug, rc::Rc};

use ndarray::prelude::*;

type ArcArrayD<A> = ArcArray<A, IxDyn>;

#[derive(Clone, Debug)]
pub struct Tensor {
    array: ArcArrayD<f64>,
    node: Option<Rc<RefCell<Node>>>,
    requires_grad: bool,
}

impl Tensor {
    fn new(array: ArrayD<f64>) -> Tensor {
        Tensor {
            array: array.into_shared(),
            node: None,
            requires_grad: false,
        }
    }

    fn require_grad(&mut self) {
        if !self.requires_grad {
            self.requires_grad = true;
            self.node = Some(Rc::new(RefCell::new(Node::new(None))));
        }
    }

    fn backward(&self) {
        if self.array.ndim() != 0 && self.array.ndim() != 1 {
            panic!("You cannot call 'backward' on a non 1-dimensional tensor");
        }

        match &self.node {
            Some(rc_node) => {
                rc_node
                    .borrow_mut()
                    .backward(&Tensor::new(ArrayD::ones(IxDyn(&[]))));
                ()
            }
            None => panic!(),
        }
    }

    fn gradient(&mut self) -> Option<Tensor> {
        if self.node.is_none() {
            return None;
        }

        let rc_node = Rc::clone(self.node.as_ref().unwrap());

        self.node = None;

        Rc::try_unwrap(rc_node)
            .ok()
            .unwrap()
            .into_inner()
            .gradient()
    }

    fn dot(&self, rhs: &Tensor) -> Tensor {
        let ndim_lhs = self.array.ndim();
        let ndim_rhs = rhs.array.ndim();

        match (ndim_lhs, ndim_rhs) {
            (0, _) | (_, 0) => Tensor::new(&self.array * &rhs.array),
            (1, 1) => Tensor::new(ArrayD::<f64>::from_elem(
                IxDyn(&[]),
                self.array
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .ok()
                    .unwrap()
                    .dot(&rhs.array.clone().into_dimensionality::<Ix1>().ok().unwrap()),
            )),
            (1, 2) => Tensor::new(
                self.array
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .ok()
                    .unwrap()
                    .dot(&rhs.array.clone().into_dimensionality::<Ix2>().ok().unwrap())
                    .into_dyn(),
            ),
            (2, 1) => Tensor::new(
                self.array
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .ok()
                    .unwrap()
                    .dot(&rhs.array.clone().into_dimensionality::<Ix1>().ok().unwrap())
                    .into_dyn(),
            ),
            (2, 2) => Tensor::new(
                self.array
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .ok()
                    .unwrap()
                    .dot(&rhs.array.clone().into_dimensionality::<Ix2>().ok().unwrap())
                    .into_dyn(),
            ),
            (_, _) => todo!(),
        }
    }

    fn square(&mut self) -> Tensor {
        call_and_build_graph_unary(Square::default(), self)
    }

    fn add(&mut self, rhs: &mut Tensor) -> Tensor {
        call_and_build_graph_binary(Add::default(), self, rhs)
    }

    fn mul(&mut self, rhs: &mut Tensor) -> Tensor {
        call_and_build_graph_binary(Mul::default(), self, rhs)
    }

    fn sum(&mut self) -> Tensor {
        call_and_build_graph_unary(Sum::default(), self)
    }
}

fn call_and_build_graph_unary<F: Function + 'static>(
    mut function: F,
    input: &mut Tensor,
) -> Tensor {
    let mut result = function.call_checked(vec![input]);

    if input.requires_grad {
        result.requires_grad = true;

        let mut node = Node::new(Some(Box::new(function)));
        node.change_children(vec![(Some(Rc::clone(&input.node.as_ref().unwrap())))]);
        result.node = Some(Rc::new(RefCell::new(node)));
    }

    result
}

fn call_and_build_graph_binary<F: Function + 'static>(
    mut function: F,
    input1: &mut Tensor,
    input2: &mut Tensor,
) -> Tensor {
    let mut result = function.call_checked(vec![input1, input2]);

    if input1.requires_grad || input2.requires_grad {
        result.requires_grad = true;

        let mut node = Node::new(Some(Box::new(function)));

        let new_children = vec![
            {
                if input1.requires_grad {
                    Some(Rc::clone(&input1.node.as_ref().unwrap()))
                } else {
                    None
                }
            },
            {
                if input2.requires_grad {
                    Some(Rc::clone(&input2.node.as_ref().unwrap()))
                } else {
                    None
                }
            },
        ];

        node.change_children(new_children);

        result.node = Some(Rc::new(RefCell::new(node)));
    }

    result
}

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
            Tensor::new(ArrayD::<f64>::ones(IxDyn(&self.shape1))).dot(outer_gradient),
            Tensor::new(ArrayD::<f64>::ones(IxDyn(&self.shape2))).dot(outer_gradient),
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
            Tensor::new(self.input2.as_ref().unwrap().to_owned()).dot(outer_gradient),
            Tensor::new(self.input1.as_ref().unwrap().to_owned()).dot(outer_gradient),
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

#[derive(Debug)]
pub enum Node {
    Leaf {
        gradient: Option<Tensor>,
    },
    Regular {
        function: Box<dyn Function>,
        children: Option<Vec<Option<Rc<RefCell<Node>>>>>,
        gradient: Option<Tensor>,
    },
}

impl Debug for dyn Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dyn Function")
    }
}

impl Node {
    fn new(function: Option<Box<dyn Function>>) -> Node {
        match function {
            Some(f) => Node::Regular {
                function: f,
                children: None,
                gradient: None,
            },
            None => Node::Leaf { gradient: None },
        }
    }

    fn gradient(self) -> Option<Tensor> {
        match self {
            Self::Leaf { gradient } => gradient,
            Self::Regular { gradient, .. } => gradient,
        }
    }

    fn reset_gradient(&mut self) {
        match self {
            Self::Leaf { gradient } => *gradient = None,
            Self::Regular { gradient, .. } => *gradient = None,
        }
    }

    fn change_children(&mut self, new_children: Vec<Option<Rc<RefCell<Node>>>>) {
        match self {
            Self::Leaf { .. } => panic!(),
            Self::Regular { children, .. } => *children = Some(new_children),
        }
    }

    fn backward(&mut self, outer_gradient: &Tensor) {
        match self {
            Self::Leaf { gradient } => {
                *gradient = match gradient {
                    Some(g) => Some(Tensor::new(&g.array + &outer_gradient.array)),
                    None => Some(outer_gradient.clone()),
                };
            }
            Self::Regular {
                function,
                children,
                gradient,
                ..
            } => {
                *gradient = match gradient {
                    Some(g) => Some(Tensor::new(&g.array + &outer_gradient.array)),
                    None => Some(outer_gradient.clone()),
                };

                let inner_gradient = function.gradient(outer_gradient);

                if let Some(c) = children {
                    for (i, child_opt) in c.iter_mut().enumerate() {
                        if let Some(child_rc) = child_opt {
                            child_rc.borrow_mut().backward(&inner_gradient[i])
                        }
                    }
                }

                *children = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Add;

    use crate::*;

    #[test]
    fn it_works() {
        let mut tensor = Tensor::new(ArrayD::<f64>::ones(IxDyn(&[2])).add(3.0));
        tensor.require_grad();

        let result = tensor.square().sum();

        result.backward();

        println!("{:#?}", tensor.gradient().unwrap());
    }
}
