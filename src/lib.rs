pub mod function;
mod ndarray_util;
mod node;
pub mod tensor;

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};

    use crate::tensor::Tensor;

    #[test]
    fn it_works() {
        let mut tensor = Tensor::new(
            ArrayD::<f64>::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        let mut tensor2 = Tensor::new(
            ArrayD::<f64>::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        tensor.require_grad();
        tensor2.require_grad();

        let mut result = tensor.matmul(&mut tensor2).mul(&mut tensor);

        result.sum().backward();

        println!("A {:#?}", tensor.array);
        println!("B {:#?}", tensor2.array);
        println!("C {:#?}", result.array);
        println!("Grad {:#?}", tensor.gradient().unwrap().array);
        println!("Grad {:#?}", tensor2.gradient().unwrap().array);
    }
}
