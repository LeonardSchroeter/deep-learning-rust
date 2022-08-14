use std::{cell::RefCell, fmt::Debug, rc::Rc};

use crate::{node::Node, tensor::Tensor};

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

#[macro_export]
macro_rules! impl_tensor_unary {
    ($($t:ty as $func_name:ident),+) => {
        impl Tensor {
            $(
                pub fn $func_name(&mut self) -> Tensor {
                    <$t>::call_and_build_graph_unary(self)
                }
            )+
        }
    };
}

#[macro_export]
macro_rules! impl_tensor_binary {
    ($($t:ty as $func_name:ident),+) => {
        impl Tensor {
            $(
                pub fn $func_name(&mut self, rhs: &mut Tensor) -> Tensor {
                    <$t>::call_and_build_graph_binary(self, rhs)
                }
            )+
        }
    };
}
