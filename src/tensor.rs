use std::{
    cell::RefCell,
    ops::{Add, AddAssign, Mul, SubAssign},
    rc::Rc,
};

use float_cmp::ApproxEq;
use ndarray::{ArrayD, Dimension, IxDyn, Zip};
use ndarray_rand::{rand_distr::Distribution, RandomExt};

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

    pub fn require_grad(self) -> Self {
        Self {
            array: self.array,
            requires_grad: true,
            node: self.node.or(Some(Rc::new(RefCell::new(Node::new(None))))),
        }
    }

    pub fn backward(&self) {
        if self.array.ndim() != 0 && self.array.ndim() != 1 {
            panic!("You cannot call 'backward' on a non 1-dimensional tensor");
        }

        match &self.node {
            Some(rc_node) => {
                rc_node.borrow_mut().backward(&Self::ones(&[]));
            }
            None => panic!(),
        }
    }

    pub fn gradient(&self) -> Option<Self> {
        if self.node.is_none() {
            return None;
        }

        self.node.as_ref().unwrap().borrow_mut().gradient()
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

    pub fn random<IdS>(shape: &[usize], distribution: IdS) -> Self
    where
        IdS: Distribution<f64>,
    {
        Self::new(ArrayD::<f64>::random(IxDyn(shape), distribution))
    }

    pub fn broadcast_backwards(&self, target_shape: Vec<usize>) -> Self {
        Tensor::new(broadcast_backwards(&self.array, target_shape))
    }

    pub fn add_in_place(&mut self, rhs: &Tensor) {
        self.array += &rhs.array;
    }

    pub fn sub_in_place(&mut self, rhs: &Tensor) {
        self.array -= &rhs.array;
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        self.plus(rhs)
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        self.plus(&Tensor::from_elem(&[], rhs))
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        self.times(rhs)
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        self.times(&Tensor::from_elem(&[], rhs))
    }
}

impl AddAssign for Tensor {
    fn add_assign(&mut self, rhs: Self) {
        self.add_in_place(&rhs)
    }
}

impl SubAssign for Tensor {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_in_place(&rhs)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.array == other.array
    }
}

impl<M: Copy + std::default::Default> ApproxEq for Tensor
where
    f64: ApproxEq<Margin = M>,
{
    type Margin = M;

    fn approx_eq<T: Into<Self::Margin>>(self, other: Self, margin: T) -> bool {
        let margin = margin.into();

        Zip::from(&self.array)
            .and(&other.array)
            .all(|a, b| a.approx_eq(*b, margin))
    }

    fn approx_ne<T: Into<Self::Margin>>(self, other: Self, margin: T) -> bool {
        !self.approx_eq(other, margin)
    }
}
