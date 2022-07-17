use core::panic;
use std::{cell::RefCell, rc::Rc};

use ndarray::prelude::*;

#[derive(Clone)]
pub struct Tensor {
    array: ArrayD<f64>,
    node: Option<Rc<RefCell<Node>>>,
    requires_grad: bool,
}

impl Tensor {
    fn new(array: ArrayD<f64>) -> Tensor {
        Tensor {
            array,
            node: None,
            requires_grad: false,
        }
    }

    fn require_grad(&mut self) {
        self.requires_grad = true;
        self.node = Some(Rc::new(RefCell::new(Node::new(None))));
    }

    fn backward(&self) {
        match &self.node {
            Some(rc_node) => {
                rc_node.borrow_mut().backward(None);
                ()
            }
            None => panic!(),
        }
    }

    fn square(&mut self) -> Tensor {
        call_and_build_graph_unary(Square { input: None }, self)
    }

    fn add(&mut self, rhs: &mut Tensor) -> Tensor {
        call_and_build_graph_binary(
            Add {
                shape1: vec![],
                shape2: vec![],
            },
            self,
            rhs,
        )
    }

    fn mul(&mut self, rhs: &mut Tensor) -> Tensor {
        call_and_build_graph_binary(
            Mul {
                input1: None,
                input2: None,
            },
            self,
            rhs,
        )
    }
}

fn call_and_build_graph_unary<F: Function + 'static>(
    mut function: F,
    input: &mut Tensor,
) -> Tensor {
    let mut result = function.call(Input::Unary(input));

    if input.requires_grad {
        result.requires_grad = true;

        let mut node = Node::new(Some(Box::new(function)));
        node.change_children(Arity::Unary(Some(Rc::clone(&input.node.as_ref().unwrap()))));
        result.node = Some(Rc::new(RefCell::new(node)));
    }

    result
}

fn call_and_build_graph_binary<F: Function + 'static>(
    mut function: F,
    input1: &mut Tensor,
    input2: &mut Tensor,
) -> Tensor {
    let mut result = function.call(Input::Binary(input1, input2));

    if input1.requires_grad || input2.requires_grad {
        result.requires_grad = true;

        let mut node = Node::new(Some(Box::new(function)));

        let new_children = Arity::Binary(
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
        );

        node.change_children(new_children);

        result.node = Some(Rc::new(RefCell::new(node)));
    }

    result
}

pub enum Arity<T> {
    Unary(T),
    Binary(T, T),
}

impl<T> Arity<T> {
    fn as_unary(self) -> T {
        match self {
            Self::Unary(element) => element,
            Self::Binary(_, _) => panic!("Not unary"),
        }
    }

    fn as_binary(self) -> (T, T) {
        match self {
            Self::Unary(_) => panic!("Not binary"),
            Self::Binary(element1, element2) => (element1, element2),
        }
    }
}

type Gradient = Arity<Tensor>;
type Input<'a> = Arity<&'a mut Tensor>;

pub trait Function {
    fn call(&mut self, input: Input) -> Tensor;

    fn backward(&self) -> Gradient;
}

pub struct Square {
    input: Option<Tensor>,
}

impl Function for Square {
    fn call(&mut self, input: Input) -> Tensor {
        let unary_input = input.as_unary();

        self.input = Some(unary_input.clone());

        Tensor::new(&unary_input.array * &unary_input.array)
    }

    fn backward(&self) -> Gradient {
        Gradient::Unary(Tensor::new(&self.input.as_ref().unwrap().array * 2.0))
    }
}

pub struct Add {
    shape1: Vec<usize>,
    shape2: Vec<usize>,
}

impl Function for Add {
    fn call(&mut self, input: Input) -> Tensor {
        let (input1, input2) = input.as_binary();

        self.shape1 = input1.array.shape().to_vec();
        self.shape2 = input2.array.shape().to_vec();

        Tensor::new(&input1.array + &input2.array)
    }

    fn backward(&self) -> Gradient {
        Gradient::Binary(
            Tensor::new(ArrayD::<f64>::ones(IxDyn(&self.shape1))),
            Tensor::new(ArrayD::<f64>::ones(IxDyn(&self.shape2))),
        )
    }
}

pub struct Mul {
    input1: Option<Tensor>,
    input2: Option<Tensor>,
}

impl Function for Mul {
    fn call(&mut self, input: Input) -> Tensor {
        let (input1, input2) = input.as_binary();

        self.input1 = Some(input1.clone());
        self.input2 = Some(input2.clone());

        Tensor::new(&input1.array * &input2.array)
    }

    fn backward(&self) -> Gradient {
        Gradient::Binary(
            self.input2.as_ref().unwrap().clone(),
            self.input1.as_ref().unwrap().clone(),
        )
    }
}

pub enum Node {
    Leaf {
        gradient: Option<Tensor>,
    },
    Regular {
        function: Box<dyn Function>,
        children: Option<Arity<Option<Rc<RefCell<Node>>>>>,
        gradient: Option<Tensor>,
    },
}

impl Node {
    fn new(function: Option<Box<dyn Function>>) -> Node {
        match function {
            Some(f) => Node::Regular {
                function: f,
                children: None,
                gradient: None,
            },
            Node => Node::Leaf { gradient: None },
        }
    }

    fn gradient(&self) -> Option<&Tensor> {
        match self {
            Self::Leaf { gradient } => gradient.as_ref(),
            Self::Regular {
                function,
                children,
                gradient,
            } => gradient.as_ref(),
        }
    }

    fn change_children(&mut self, new_children: Arity<Option<Rc<RefCell<Node>>>>) {
        match self {
            Self::Leaf { gradient } => panic!(),
            Self::Regular { children, .. } => *children = Some(new_children),
        }
    }

    fn backward(&mut self, outer_gradient: Option<&Tensor>) {
        match self {
            Self::Leaf { gradient } => match outer_gradient {
                Some(outer_grad) => {
                    *gradient = match gradient {
                        Some(g) => Some(Tensor::new(&g.array + &outer_grad.array)),
                        None => Some(outer_grad.clone()),
                    };
                }
                None => (),
            },
            Self::Regular {
                function,
                children,
                gradient,
            } => {
                let inner_gradient = function.backward();
                let result = match outer_gradient {
                    Some(outer_grad) => {
                        *gradient = match gradient {
                            Some(g) => Some(Tensor::new(&g.array + &outer_grad.array)),
                            None => Some(outer_grad.clone()),
                        };

                        match inner_gradient {
                            Arity::Unary(tensor) => {
                                Arity::Unary(Tensor::new(&tensor.array * &outer_grad.array))
                            }
                            Arity::Binary(tensor1, tensor2) => Arity::Binary(
                                Tensor::new(&tensor1.array * &outer_grad.array),
                                Tensor::new(&tensor2.array * &outer_grad.array),
                            ),
                        }
                    }
                    None => inner_gradient,
                };

                if let Some(c) = children {
                    match c {
                        Arity::Unary(child_opt) => {
                            if let Some(child_rc) = child_opt {
                                child_rc.borrow_mut().backward(Some(&result.as_unary()))
                            }
                        }
                        Arity::Binary(child_opt1, child_opt2) => {
                            let result = result.as_binary();
                            if let Some(child_rc) = child_opt1 {
                                child_rc.borrow_mut().backward(Some(&result.0))
                            }
                            if let Some(child_rc) = child_opt2 {
                                child_rc.borrow_mut().backward(Some(&result.1))
                            }
                        }
                    }
                }
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
        let result = 2 + 2;
        assert_eq!(result, 4);

        let mut tensor = Tensor::new(ArrayD::<f64>::ones(IxDyn(&[1])).add(3.0));
        tensor.require_grad();

        let mut result = tensor.square().mul(&mut tensor).square();
        result.backward();

        println!("{:#?}", result.array);
        println!(
            "{:#?}",
            tensor.node.unwrap().borrow().gradient().unwrap().array
        );
    }
}
