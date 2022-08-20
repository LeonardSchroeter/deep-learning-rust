pub mod function;
pub mod functions;
mod ndarray_util;
mod node;
pub mod tensor;

#[cfg(test)]
mod tests {
    use mnist::{Mnist, MnistBuilder};
    use ndarray::s;

    use crate::tensor::Tensor;

    #[test]
    fn neural_net() {
        // env::set_var("RUST_BACKTRACE", "1");

        let Mnist {
            trn_img, trn_lbl, ..
        } = MnistBuilder::new()
            .label_format_one_hot()
            .training_set_length(50_000)
            .validation_set_length(10_000)
            .test_set_length(10_000)
            .finalize();

        let mut x = Tensor::from_shape_vec(
            &[50_000, 784],
            trn_img.into_iter().map(|x| x as f64 / 256.).collect(),
        );

        let mut y = Tensor::from_shape_vec(
            &[50_000, 10],
            trn_lbl.into_iter().map(|y| y as f64).collect(),
        );

        const INPUT_DIM: usize = 784;
        const LAYER_DIM: usize = 10;

        // let mut x = Tensor::from_shape_vec(&[INPUT_DIM], vec![1.0, 2.0]);
        // let mut y = Tensor::from_shape_vec(&[LAYER_DIM], vec![1.0, 2.0]);

        let mut weights = Tensor::from_elem(&[INPUT_DIM, LAYER_DIM], 0.01);
        weights.require_grad();
        let mut bias = Tensor::from_elem(&[LAYER_DIM], 0.01);
        bias.require_grad();

        for i in 1..1000 {
            let mut x1 = x.matmul(&mut weights);
            let mut x2 = x1.add(&mut bias);
            let mut x3 = x2.relu();

            let loss = x3.mse(&mut y);

            if i % 10 == 0 {
                println!("y {:#?}", y.array.slice(s![0..10, ..]));
                println!("out {:#?}", x3.array.slice(s![0..10, ..]));
            }
            println!("Loss {:#?}", loss.array);

            loss.backward();

            let weights_gradient = weights.gradient().unwrap();
            weights.array -= &(&weights_gradient.array * 0.01);

            let bias_gradient = bias.gradient().unwrap();

            bias.array -= &(&bias_gradient.array * 0.01);
        }
    }
}
