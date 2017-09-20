#include "ann.h"
#include "helper.h"

int main() {
  
  // load date
  MatrixXd xtrain(1383, 12), ytrain(1383, 1);  // 1383 training examples
  MatrixXd xval(592, 12), yval(592,1);  // 592 validation examples
  load_data("../data/preprocessed/training.txt", xtrain, ytrain);
  load_data("../data/preprocessed/validation.txt", xval, yval);
  
  VectorXi layer_size(3);
  layer_size << 12, 12, 4;
  
  VectorXi activation(2);
  activation << Sigmoid, Sigmoid;
  
  ANN ann;
  ann.set_arch(layer_size, activation, CrossEntropy);
  ann.set_weights(Uniform);
  ann.set_grad_desc(400, Batch);
  ann.set_regulariz(L2, 0.03);
  ann.set_lr(1);
  ann.train(xtrain, ytrain, 1);
  
  MxPtr eval = nullptr;
  ann.evaluate_classify(xval, yval, eval);
  print_eval_classify(eval);
  
  
  
  // An example of 3 layer neural network training with techniques
  // VectorXi layer_size2(4);
  // layer_size2 << 12, 12, 12, 4;
  
  // VectorXi activation2(3);
  // activation2 << Sigmoid, Sigmoid, Sigmoid;
  
  // ANN ann2;
  // ann2.set_arch(layer_size2, activation2, CrossEntropy);
  // ann2.set_weights(Glorot);
  // ann2.set_grad_desc(400, MiniBatch, 1400);
  // ann2.set_regulariz(L2, 0.03);
  // ann2.set_lr(1, Scal, 200, Epoch);
  // ann2.set_momentum(LinInc, 0.5, 0.9, 200, 300, Epoch);
  // ann2.set_early_stop(ON, 10);
  // ann2.train_val(xtrain, ytrain, xval, yval, 1);
  
  // MxPtr eval2 = nullptr;
  // ann2.evaluate_classify(xval, yval, eval2);
  // print_eval_classify(eval2);
  
  // vector<double> * cost_train = ann2.get_train_cost();
  // vector<double> * cost_val = ann2.get_val_cost();
  // vector<double> * ce_train = ann2.get_train_ce();
  // vector<double> * ce_val = ann2.get_val_ce();
  
  // save_cost("cost_train.txt", cost_train);
  // save_cost("cost_val.txt", cost_val);
  // save_cost("ce_train.txt", ce_train);
  // save_cost("ce_val.txt", ce_val);
  
  return 0;
}
