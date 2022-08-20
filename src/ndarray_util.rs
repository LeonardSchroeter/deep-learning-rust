use ndarray::{ArcArray, ArrayD, Axis, IxDyn};

pub type ArcArrayD<A> = ArcArray<A, IxDyn>;

pub fn broadcast_backwards(input: &ArcArrayD<f64>, target_shape: Vec<usize>) -> ArrayD<f64> {
    let mut axes: Vec<usize> = Vec::new();

    let mut target_shape_iter = target_shape.iter();
    let mut cur_dim = *target_shape_iter.next_back().unwrap_or(&0);
    for (axis, &result_dim) in input.shape().to_vec().iter().enumerate().rev() {
        if cur_dim == result_dim {
            cur_dim = *target_shape_iter.next_back().unwrap_or(&0);
        } else {
            axes.push(axis);
        }
    }

    let mut result = input.to_owned();
    for &axis in axes.iter() {
        result = result.sum_axis(Axis(axis));
    }

    result.into_shape(target_shape).expect("Wrong shape")
}
