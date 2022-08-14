pub mod function;
pub mod functions;
mod ndarray_util;
mod node;
pub mod tensor;

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn neural_net() {
        const INPUT_DIM: usize = 2;
        const LAYER_DIM: usize = 2;

        let mut x = Tensor::from_shape_vec(&[INPUT_DIM], vec![1.0, 2.0]);
        let mut y = Tensor::from_shape_vec(&[LAYER_DIM], vec![1.0, 2.0]);

        let mut weights = Tensor::ones(&[INPUT_DIM, LAYER_DIM]);
        weights.require_grad();
        let mut bias = Tensor::ones(&[LAYER_DIM]);
        bias.require_grad();

        for _ in 1..1000 {
            let mut x1 = x.matmul(&mut weights);
            let mut x2 = x1.add(&mut bias);
            let mut x3 = x2.relu();

            let loss = x3.mse(&mut y);

            println!("x {:#?}", x.array);
            println!("y {:#?}", y.array);
            println!("x3 {:#?}", x3.array);
            println!("Loss {:#?}", loss.array);

            loss.backward();

            let weights_gradient = weights.gradient().unwrap();
            weights.array -= &(&weights_gradient.array * 0.001);

            let bias_gradient = bias.gradient().unwrap();
            bias.array -= &(&bias_gradient.array * 0.001);
        }
    }
}
