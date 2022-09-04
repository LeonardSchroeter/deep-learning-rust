use deep_learning::{
    layer::{Layer, Linear},
    optimizer::{Optimizer, SGD},
    tensor::Tensor,
};
use mnist::{Mnist, MnistBuilder};

pub struct MyNeuralNet {
    input_layer: Linear,
    hidden_layer: Linear,
    output_layer: Linear,
}

impl MyNeuralNet {
    fn new() -> Self {
        Self {
            input_layer: Linear::new(784, 128, Some(Tensor::sigmoid)),
            hidden_layer: Linear::new(128, 64, Some(Tensor::sigmoid)),
            output_layer: Linear::new(64, 10, None),
        }
    }
}

impl Layer for MyNeuralNet {
    fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.input_layer.forward(input);
        let x = self.hidden_layer.forward(&x);
        self.output_layer.forward(&x)
    }

    fn trainable_weights(&mut self) -> Vec<&mut Tensor> {
        self.input_layer
            .trainable_weights()
            .into_iter()
            .chain(self.hidden_layer.trainable_weights().into_iter())
            .chain(self.output_layer.trainable_weights().into_iter())
            .collect()
    }
}

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

    let mut nn = MyNeuralNet::new();

    let optimizer = SGD::new(0.01);

    for epoch in 1..EPOCHS {
        for (i, (x, y)) in x.array.outer_iter().zip(y.array.outer_iter()).enumerate() {
            let x = Tensor::new(x.to_owned());
            let y = Tensor::new(y.to_owned());

            let y_pred = nn.forward(&x);

            let loss = y_pred.cross_entropy_with_softmax(&y);

            if i % 100 == 0 {
                // println!("true \n{:?}", y.array.slice(s![0..10, ..]));
                // println!("out \n{:?}", y_pred.array.slice(s![0..10, ..]));
                println!("Epoch {}: Loss {:#?}", epoch, loss.array);
            }

            loss.backward();

            optimizer.step(nn.trainable_weights())
        }
    }
}
