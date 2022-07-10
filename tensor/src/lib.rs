use enum_as_inner::EnumAsInner;

#[derive(Clone, Debug, EnumAsInner)]
pub enum NestedArray<T> {
    Base(Vec<T>),
    Nested(Vec<NestedArray<T>>),
}

impl<T: Default + std::clone::Clone> NestedArray<T> {
    fn new<Iter: IntoIterator<Item = usize>>(shape: Iter) -> Self
    where
        Iter::IntoIter: DoubleEndedIterator,
    {
        let mut iter = shape.into_iter().rev().into_iter();
        let innermost_dim = iter.next().unwrap_or(1);
        let mut nested_array = NestedArray::<T>::Base(vec![T::default(); innermost_dim]);

        for dim in iter {
            nested_array = NestedArray::<T>::Nested(vec![nested_array.clone(); dim]);
        }

        nested_array
    }

    fn to_vec(&self) -> Vec<T> {
        match self {
            NestedArray::<T>::Base(v) => v.clone(),
            NestedArray::<T>::Nested(v) => v
                .iter()
                .map(|nested_array| nested_array.to_vec())
                .flatten()
                .collect(),
        }
    }
}

pub struct Tensor {
    values: NestedArray<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    fn new(shape: Vec<usize>, values: Option<NestedArray<f32>>) -> Tensor {
        Tensor {
            values: match values {
                None => NestedArray::<f32>::new(shape.clone()),
                Some(values) => values,
            },
            shape: shape.clone(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type NestedArrayFloat = NestedArray<f32>;

    #[test]
    fn nested_base() {
        let nested_array = NestedArrayFloat::new([1]);
        assert_eq!(nested_array.into_base().unwrap(), vec![0.0]);
    }

    #[test]
    fn nested_two_dim() {
        let nested_array = NestedArrayFloat::new([2, 3]);
        println!("{:#?}", nested_array);
        assert_eq!(nested_array.to_vec(), vec![0.0; 6]);
        let unwrapped_vec = nested_array.into_nested().unwrap();
        assert_eq!(unwrapped_vec[0].clone().into_base().unwrap(), vec![0.0; 3]);
        assert_eq!(unwrapped_vec[1].clone().into_base().unwrap(), vec![0.0; 3]);
    }
}
