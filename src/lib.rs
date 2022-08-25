pub mod function;
pub mod functions;
pub mod layer;
mod ndarray_util;
mod node;
pub mod optimizer;
pub mod tensor;

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn some_test() {
        let input = Tensor::from_shape_vec(&[2, 2], vec![1., 2., 4., 5.]);
        let output = Tensor::from_shape_vec(&[2, 2], vec![1., 0., 0., 1.]);
        let weights = Tensor::from_shape_vec(&[2, 2], vec![0.1, 0.2, 0.3, 0.4]).require_grad();
        let bias = Tensor::from_shape_vec(&[1], vec![0.4]).require_grad();

        let out = input.matmul(&weights).plus(&bias);
        let l = out.cross_entropy_with_softmax(&output);

        println!("{:?}", l);
    }
}
