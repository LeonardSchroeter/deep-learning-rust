use std::{cell::RefCell, rc::Rc};

use ndarray::{ArrayD, Dimension, Ix1, Ix2, IxDyn};

use crate::{
    ndarray_util::{broadcast_backwards, ArcArrayD},
    node::Node,
};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub array: ArcArrayD<f64>,
    pub node: Option<Rc<RefCell<Node>>>,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn new(array: ArrayD<f64>) -> Self {
        Self {
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
                    .backward(&Self::new(ArrayD::ones(IxDyn(&[]))));
                ()
            }
            None => panic!(),
        }
    }

    pub fn gradient(&mut self) -> Option<Self> {
        if self.node.is_none() {
            return None;
        }

        let rc_node = Rc::clone(self.node.as_ref().unwrap());

        self.node = Some(Rc::new(RefCell::new(Node::new(None))));

        Rc::try_unwrap(rc_node)
            .ok()
            .unwrap()
            .into_inner()
            .gradient()
    }

    pub fn dot(&self, rhs: &Self) -> Self {
        let ndim_lhs = self.array.ndim();
        let ndim_rhs = rhs.array.ndim();

        match (ndim_lhs, ndim_rhs) {
            (0, _) | (_, 0) => Self::new(&self.array * &rhs.array),
            (1, 1) => Self::new(ArrayD::<f64>::from_elem(
                IxDyn(&[]),
                self.array
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .ok()
                    .unwrap()
                    .dot(&rhs.array.clone().into_dimensionality::<Ix1>().ok().unwrap()),
            )),
            (1, 2) => Self::new(
                self.array
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .ok()
                    .unwrap()
                    .dot(&rhs.array.clone().into_dimensionality::<Ix2>().ok().unwrap())
                    .into_dyn(),
            ),
            (2, 1) => Self::new(
                self.array
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .ok()
                    .unwrap()
                    .dot(&rhs.array.clone().into_dimensionality::<Ix1>().ok().unwrap())
                    .into_dyn(),
            ),
            (2, 2) => Self::new(
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

    pub fn zeros(shape: &[usize]) -> Self {
        Self::new(ArrayD::<f64>::zeros(IxDyn(shape)))
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::new(ArrayD::<f64>::ones(IxDyn(shape)))
    }

    pub fn from_elem(shape: &[usize], elem: f64) -> Self {
        Self::new(ArrayD::<f64>::from_elem(IxDyn(shape), elem))
    }

    pub fn from_shape_vec(shape: &[usize], v: Vec<f64>) -> Self {
        Self::new(ArrayD::<f64>::from_shape_vec(IxDyn(shape), v).unwrap())
    }

    pub fn from_shape_simple_fn<F>(shape: &[usize], f: F) -> Self
    where
        F: FnMut() -> f64,
    {
        Self::new(ArrayD::<f64>::from_shape_simple_fn(IxDyn(shape), f))
    }

    pub fn from_shape_fn<F>(shape: &[usize], f: F) -> Self
    where
        F: FnMut(<IxDyn as Dimension>::Pattern) -> f64,
    {
        Self::new(ArrayD::<f64>::from_shape_fn(IxDyn(shape), f))
    }

    pub fn broadcast_backwards(&self, target_shape: Vec<usize>) -> Self {
        Tensor::new(broadcast_backwards(&self.array, target_shape))
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.array == other.array
    }
}
