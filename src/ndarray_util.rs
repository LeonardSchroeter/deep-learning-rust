use ndarray::{ArcArray, ArrayBase, ArrayD, Axis, Data, IxDyn};

pub type ArcArrayD<A> = ArcArray<A, IxDyn>;

pub fn broadcast_backwards<A: Data<Elem = f64>>(
    input: &ArrayBase<A, IxDyn>,
    target_shape: Vec<usize>,
) -> ArrayD<f64> {
    let mut axes: Vec<usize> = Vec::new();

    let mut target_shape_iter = target_shape.iter();
    let mut cur_dim = *target_shape_iter.next_back().unwrap_or(&0);
    for (axis, &result_dim) in input.shape().iter().enumerate().rev() {
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

pub fn arg_max<A: Data<Elem = f64>>(array: &ArrayBase<A, IxDyn>) -> usize {
    let mut result = f64::NEG_INFINITY;
    let mut index = 0;

    for (i, &value) in array.iter().enumerate() {
        if value > result {
            result = value;
            index = i;
        }
    }

    index
}
