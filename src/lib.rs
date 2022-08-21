pub mod function;
pub mod functions;
mod ndarray_util;
mod node;
pub mod tensor;

#[cfg(test)]
mod tests {
    use mnist::{Mnist, MnistBuilder};
    use ndarray::s;
    use ndarray_rand::rand_distr::StandardNormal;

    use crate::tensor::Tensor;

    #[test]
    fn some_test() {
        let mut input = Tensor::from_shape_vec(&[2, 2], vec![1., 2., 4., 5.]);
        let mut output = Tensor::from_shape_vec(&[2, 2], vec![1., 0., 0., 1.]);
        let mut weights = Tensor::from_shape_vec(&[2, 2], vec![0.1, 0.2, 0.3, 0.4]);
        let mut bias = Tensor::from_shape_vec(&[1], vec![0.4]);
        weights.require_grad();
        bias.require_grad();

        let mut out = input.matmul(&mut weights).add(&mut bias);
        let l = out.cross_entropy_with_softmax(&mut output);

        println!("{:?}", l);
    }

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
            &[500, 100, 784],
            trn_img.into_iter().map(|x| x as f64 / 256.).collect(),
        );

        let mut y = Tensor::from_shape_vec(
            &[500, 100, 10],
            trn_lbl.into_iter().map(|y| y as f64).collect(),
        );

        const EPOCHS: usize = 10;

        const INPUT_DIM: usize = 784;
        const HIDDEN_DIM_1: usize = 128;
        const HIDDEN_DIM_2: usize = 64;
        const OUTPUT_DIM: usize = 10;

        let mut w1 = Tensor::random(&[INPUT_DIM, HIDDEN_DIM_1], StandardNormal);
        w1.require_grad();
        let mut w2 = Tensor::random(&[HIDDEN_DIM_1, HIDDEN_DIM_2], StandardNormal);
        w2.require_grad();
        let mut w3 = Tensor::random(&[HIDDEN_DIM_2, OUTPUT_DIM], StandardNormal);
        w3.require_grad();
        let mut b1 = Tensor::random(&[HIDDEN_DIM_1], StandardNormal);
        b1.require_grad();
        let mut b2 = Tensor::random(&[HIDDEN_DIM_2], StandardNormal);
        b2.require_grad();
        let mut b3 = Tensor::random(&[OUTPUT_DIM], StandardNormal);
        b3.require_grad();

        for epoch in 1..EPOCHS {
            for (i, (x, y)) in x
                .array
                .outer_iter_mut()
                .zip(y.array.outer_iter_mut())
                .enumerate()
            {
                let mut x = Tensor::new(x.to_owned());
                let mut y = Tensor::new(y.to_owned());

                let mut h1 = x.matmul(&mut w1).add(&mut b1).sigmoid();
                let mut h2 = h1.matmul(&mut w2).add(&mut b2).sigmoid();
                let mut y_pred = h2.matmul(&mut w3).add(&mut b3);

                let loss = y_pred.cross_entropy_with_softmax(&mut y);

                if i % 100 == 0 {
                    println!("true \n{:?}", y.array.slice(s![0..10, ..]));
                    println!("out \n{:?}", y_pred.array.slice(s![0..10, ..]));
                }

                println!("Epoch {}: Loss {:#?}", epoch, loss.array);

                loss.backward();

                let w1_gradient = w1.gradient().unwrap();
                w1.array -= &(&w1_gradient.array * 0.01);

                let w2_gradient = w2.gradient().unwrap();
                w2.array -= &(&w2_gradient.array * 0.01);

                let w3_gradient = w3.gradient().unwrap();
                w3.array -= &(&w3_gradient.array * 0.01);

                let b1_gradient = b1.gradient().unwrap();
                b1.array -= &(&b1_gradient.array * 0.01);

                let b2_gradient = b2.gradient().unwrap();
                b2.array -= &(&b2_gradient.array * 0.01);

                let b3_gradient = b3.gradient().unwrap();
                b3.array -= &(&b3_gradient.array * 0.01);
            }
        }
    }
}
