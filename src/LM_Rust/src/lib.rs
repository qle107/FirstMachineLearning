extern crate ndarray;
extern crate rand;

use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int};
use std::ptr;

#[repr(C)]
pub struct LinearModel {
    nb_class: usize,
    weights: Array2<f64>,
}

#[no_mangle]
pub extern "C" fn linear_model_new(nb_class: usize, nb_features: usize) -> *mut LinearModel {
    let weights = Array2::<f64>::random((nb_class, nb_features + 1), rand::distributions::Uniform::new(-0.01, 0.01));
    let model = LinearModel { nb_class, weights };
    Box::into_raw(Box::new(model))
}

#[no_mangle]
pub extern "C" fn linear_model_free(model: *mut LinearModel) {
    if !model.is_null() {
        unsafe {
            Box::from_raw(model);
        }
    }
}

#[no_mangle]
pub extern "C" fn linear_model_predict(model: *mut LinearModel, input: *const c_double, length: c_int) -> *mut c_double {
    let model = unsafe {
        assert!(!model.is_null());
        &mut *model
    };
    let input = unsafe {
        assert!(!input.is_null());
        std::slice::from_raw_parts(input, length as usize)
    };
    let input = Array1::from_vec(input.to_vec());
    let prediction = model.predict(&input);

    let mut output = Vec::with_capacity(prediction.len());
    output.extend_from_slice(prediction.as_slice().unwrap());

    let output_ptr = output.as_mut_ptr();
    std::mem::forget(output); // Prevent Vec from being deallocated
    output_ptr
}

#[no_mangle]
pub extern "C" fn linear_model_train(model: *mut LinearModel, X: *const c_double, Y: *const c_double, n_samples: i64, n_features: i64, lr: c_double, iterations: c_int) {
    let model = unsafe {
        assert!(!model.is_null());
        &mut *model
    };
    let X = unsafe {
        assert!(!X.is_null());
        Array2::from_shape_vec((n_samples as usize, n_features as usize), std::slice::from_raw_parts(X, (n_samples * n_features) as usize).to_vec()).unwrap()
    };
    let Y = unsafe {
        assert!(!Y.is_null());
        Array2::from_shape_vec((n_samples as usize, model.nb_class), std::slice::from_raw_parts(Y, (n_samples as usize * model.nb_class) as usize).to_vec()).unwrap()
    };

    model.train(&X, &Y, lr, iterations as usize);
}

impl LinearModel {
    fn predict(&self, X: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::<f64>::zeros(self.nb_class);
        for class_idx in 0..self.nb_class {
            let mut sum = self.weights[(class_idx, 0)];
            for i in 0..X.len() {
                sum += self.weights[(class_idx, i + 1)] * X[i];
            }
            scores[class_idx] = sum;
        }
        let max_index = scores.iter().enumerate().fold(0, |max_idx, (i, &val)| if val > scores[max_idx] { i } else { max_idx });
        Array1::from_vec((0..self.nb_class).map(|i| if i == max_index { 1.0 } else { 0.0 }).collect())
    }

    fn train(&mut self, X: &Array2<f64>, Y: &Array2<f64>, lr: f64, iteration: usize) {
        let mut rng = rand::thread_rng();
        for class_idx in 0..self.nb_class {
            let binary_Y: Array1<i32> = Y.column(class_idx).map(|&x| if x == 1.0 { 1 } else { -1 });
            for iter in 0..iteration {
                for _ in 0..X.nrows() {
                    let k = rng.gen_range(0..X.nrows());
                    let inp = X.row(k);
                    let predicted = self._predict_class(&inp, class_idx);
                    let target = binary_Y[k] as f64;
                    let error = target - predicted;
                    for i in 0..inp.len() {
                        self.weights[(class_idx, i + 1)] += lr * error * inp[i];
                    }
                 }
            }
        }
    }

    fn _predict_class(&self, X: &ArrayView1<f64>, class_idx: usize) -> f64 {
        let mut sum = self.weights[(class_idx, 0)];
        for i in 0..X.len() {
            sum += self.weights[(class_idx, i + 1)] * X[i];
        }
        if sum >= 0.0 { 1.0 } else { -1.0 }
    }
}
