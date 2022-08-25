use deep_learning::{
    layer::{Layer, Linear},
    optimizer::{Optimizer, SGD},
    tensor::Tensor,
};
use mnist::{Mnist, MnistBuilder};

fn main() {
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let x = Tensor::from_shape_vec(
        &[500, 100, 784],
        trn_img.into_iter().map(|x| x as f64 / 256.).collect(),
    );

    let y = Tensor::from_shape_vec(
        &[500, 100, 10],
        trn_lbl.into_iter().map(|y| y as f64).collect(),
    );

    const EPOCHS: usize = 10;

    const INPUT_DIM: usize = 784;
    const HIDDEN_DIM_1: usize = 128;
    const HIDDEN_DIM_2: usize = 64;
    const OUTPUT_DIM: usize = 10;

    let mut layer1 = Linear::new(INPUT_DIM, HIDDEN_DIM_1, Some(Tensor::sigmoid));
    let mut layer2 = Linear::new(HIDDEN_DIM_1, HIDDEN_DIM_2, Some(Tensor::sigmoid));
    let mut layer3 = Linear::new(HIDDEN_DIM_2, OUTPUT_DIM, None);

    let optimizer = SGD::new(0.01);

    for epoch in 1..EPOCHS {
        for (i, (x, y)) in x.array.outer_iter().zip(y.array.outer_iter()).enumerate() {
            let x = Tensor::new(x.to_owned());
            let y = Tensor::new(y.to_owned());

            let h1 = layer1.forward(&x);
            let h2 = layer2.forward(&h1);
            let y_pred = layer3.forward(&h2);

            let loss = y_pred.cross_entropy_with_softmax(&y);

            if i % 100 == 0 {
                // println!("true \n{:?}", y.array.slice(s![0..10, ..]));
                // println!("out \n{:?}", y_pred.array.slice(s![0..10, ..]));
                println!("Epoch {}: Loss {:#?}", epoch, loss.array);
            }

            loss.backward();
            optimizer.step(
                layer1
                    .trainable_weights()
                    .into_iter()
                    .chain(layer2.trainable_weights().into_iter())
                    .chain(layer3.trainable_weights().into_iter())
                    .collect(),
            )
        }
    }
}
