extern crate ndarray;
extern crate rand;

use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::Rng;

struct LinearModel {
    nb_class: usize,
    weights: Array2<f64>,
}

impl LinearModel {
    fn new(nb_class: usize, nb_features: usize) -> Self {
        let weights = Array2::<f64>::zeros((nb_class, nb_features + 1));
        LinearModel { nb_class, weights }
    }

    fn predict(&self, X: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::<f64>::zeros(self.nb_class);
        for class_idx in 0..self.nb_class {
            let mut sum = self.weights[(class_idx, 0)];
            for i in 0..X.len() {
                sum += self.weights[(class_idx, i + 1)] * X[i];
            }
            scores[class_idx] = sum;
        }
        let max_index = scores
            .iter()
            .enumerate()
            .fold(0, |max_idx, (i, &val)| if val > scores[max_idx] { i } else { max_idx });
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
                    self.weights[(class_idx, 0)] += lr * error;
                }

                // Calculate and print the loss every 100 iterations
                if iter % 100 == 0 {
                    let loss = self.calculate_hinge_loss(X, Y, class_idx);
                    println!("Iteration: {}, Class: {}, Loss: {}", iter, class_idx, loss);
                }
            }
        }
    }

    fn calculate_hinge_loss(&self, X: &Array2<f64>, Y: &Array2<f64>, class_idx: usize) -> f64 {
        let binary_Y: Array1<i32> = Y.column(class_idx).map(|&x| if x == 1.0 { 1 } else { -1 });
        let mut total_loss = 0.0;
        for (i, sample) in X.outer_iter().enumerate() {
            let target = binary_Y[i] as f64;
            let score = self._predict_class(&sample, class_idx);
            total_loss += (1.0 - target * score).max(0.0);
        }
        total_loss / X.nrows() as f64
    }

    fn _predict_class(&self, X: &ArrayView1<f64>, class_idx: usize) -> f64 {
        let mut sum = self.weights[(class_idx, 0)];
        for i in 0..X.len() {
            sum += self.weights[(class_idx, i + 1)] * X[i];
        }
        if sum >= 0.0 { 1.0 } else { -1.0 }
    }
}

fn main() {
    // Example usage
    let X: Array2<f64> = ndarray::array![
        [1.0, 1.0],
        [2.0, 3.0],
        [3.0, 3.0],
    ];

    let Y: Array2<f64> = ndarray::array![
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ];

    let mut model = LinearModel::new(2, X.ncols());
    let learning_rate = 0.1;
    let iterations = 1000;

    model.train(&X, &Y, learning_rate, iterations);

    let test_input = vec![ndarray::array![1.0, 1.0], ndarray::array![2.0, 3.0], ndarray::array![3.0, 3.0], ndarray::array![1.0, 1.0]];
    for i in 0..test_input.len(){
        let prediction = model.predict(&test_input[i]);
        println!("Prediction: {:?} {}", prediction, i);
    }

}
