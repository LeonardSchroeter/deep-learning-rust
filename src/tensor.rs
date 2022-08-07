use std::{cell::RefCell, rc::Rc};

use ndarray::{ArrayD, Ix1, Ix2, IxDyn};

use crate::{ndarray_util::ArcArrayD, node::Node};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub array: ArcArrayD<f64>,
    pub node: Option<Rc<RefCell<Node>>>,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn new(array: ArrayD<f64>) -> Tensor {
        Tensor {
            array: array.into_shared(),
            node: None,
            requires_grad: false,
        }
    }

    pub fn require_grad(&mut self) {
        if !self.requires_grad {
            self.requires_grad = true;
            self.node = Some(Rc::new(RefCell::new(Node::new(None))));
        }
    }

    pub fn backward(&self) {
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

    pub fn gradient(&mut self) -> Option<Tensor> {
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

    pub fn dot(&self, rhs: &Tensor) -> Tensor {
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
}
