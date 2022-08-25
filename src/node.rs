use std::{cell::RefCell, rc::Rc};

use crate::{function::Function, tensor::Tensor};

#[derive(Debug)]
pub struct Node {
    function: Option<Box<dyn Function>>,
    children: Vec<Option<Rc<RefCell<Node>>>>,
    gradient: Option<Tensor>,
}

impl Node {
    pub fn new(function: Option<Box<dyn Function>>) -> Self {
        Self {
            function,
            children: vec![],
            gradient: None,
        }
    }

    pub fn gradient(&mut self) -> Option<Tensor> {
        self.gradient.take()
    }

    pub fn change_children(&mut self, new_children: Vec<Option<Rc<RefCell<Node>>>>) {
        self.children = new_children;
    }

    pub fn backward(&mut self, outer_gradient: &Tensor) {
        self.gradient = match &self.gradient {
            Some(g) => Some(Tensor::new(&g.array + &outer_gradient.array)),
            None => Some(outer_gradient.clone()),
        };

        if self.function.is_none() {
            return;
        }

        let inner_gradient = self.function.as_ref().unwrap().gradient(outer_gradient);

        for (child, inner_gradient) in self.children.iter_mut().zip(inner_gradient) {
            if let Some(child) = child {
                child.borrow_mut().backward(&inner_gradient);
            }
        }

        self.children.clear();
    }
}
