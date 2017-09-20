#ifndef ANN_H
#define ANN_H

#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

enum Activation {Linear, Sigmoid};
enum CostFunction {MeanSquaredError, CrossEntropy};
enum WeightInit {Uniform, Glorot, He};
enum GradDesc {Stochastic, Batch, MiniBatch};
enum LRDecay {NO_Decay, EpochThres, Scal, Halve, Step};
enum Moment {NO_Mmt, LinInc};
enum Regularize {NO_Reg, L1, L2};
enum EpochIter {Epoch, Iter}; // type of lr_decay and momentum threshold
enum EarlyStop {OFF, ON};

typedef MatrixXd* MxPtr;

struct Arch {
  int num; // number of layer (input layer not included)
  int* size;
  Activation* af;
  CostFunction cf;
};

struct Weight {
  WeightInit type;
  MxPtr init;
  MxPtr train;
};

struct GradientDescent {
  GradDesc type;
  int batch_size;
  int max_epoch;
  std::vector<double>* cost_train;
  std::vector<double>* cost_val;
  std::vector<double>* ce_train;
  std::vector<double>* ce_val;
};

struct LearningRate {
  double alpha;
  LRDecay type;
  int epoch_thres;
  EpochIter ei;
};

struct Momentum {
  Moment type;
  double init;
  double final;
  int epoch_low;
  int epoch_upp;
  EpochIter ei;
};

struct Regularization {
  Regularize type;
  double lambda;
};

struct EarlyStopping {
  EarlyStop type;
  int stop_thres;
};


class ANN {
  
 private:
  
  Arch layer;
  Weight weight;
  GradientDescent gd;
  LearningRate lr;
  Momentum mmt;
  Regularization reg;
  EarlyStopping estp;
  
  // setup auxiliary functions
  void assign_new_mx(MxPtr & ptr, const int & num);
  bool check_arch(const VectorXi & _layer_size, 
		  const VectorXi & _activ_func, 
		  const CostFunction & _cost_func);
  bool check_weights(const WeightInit & _type);
  bool check_grad_desc(const int & _max_epoch, const GradDesc & _type);
  bool check_grad_desc(const int & _max_epoch, const GradDesc & _type, 
		       const int & _batch_size);
  bool check_lr(const double & _alpha, const LRDecay & _type, 
		const int & _epoch_thres, const EpochIter & _ei);
  bool check_momentum(const Moment & _type);
  bool check_momentum(const Moment & _type, const double & _init, 
		      const double & _final, const int & _epoch_low, 
		      const int & _epoch_upp, const EpochIter & _ei);
  bool check_regulariz(const Regularize & _type);
  bool check_regulariz(const Regularize & _type, const double & _lambda);
  bool check_early_stop(EarlyStop _type, const int & _stop_thres);
  
  
  void delete_arch();
  void delete_weights();
  void delete_cost();
  
  void init_weights();
  void init_weights_uniform();
  void init_weights_gaussian();
  
  
  // train
  void train_nn(const MatrixXd & x, const MatrixXd & y,
		const MatrixXd & xval, const MatrixXd & yval,
		const bool & val, const int & _type);
  
  // train auxiliary functions
  bool check_dataset(const MatrixXd & x, const MatrixXd & y, const int & _type);
  void reset_params(const MatrixXd & x);
  void index_rand_shuffle(int* & _ind, const int & _num);
  static int rand_aux(const int & i) {return std::rand()%i;};
  MatrixXd map_vectors(const MatrixXd & y);
  MatrixXd map_labels(const MatrixXd & result);
  void split_minibatch(MxPtr & _x, MxPtr & _y, MxPtr & _y_ce, 
		       const int & _num_sub, const MatrixXd & x, const MatrixXd & y, 
		       const MatrixXd & y_ce, int* _ind);
  void feedforward(MxPtr & _az, const int & _num_az, const MatrixXd & x);
  void backpropagate(MxPtr & _weight_grad, const int & _num_wg, const MxPtr & _az, 
		     const int & _num_az, const MatrixXd & y);
  
  
  // activation and cost functions and gradient functions
  MatrixXd activate(const Activation & type, const MatrixXd & z);
  MatrixXd activ_grad(const Activation & type, const MatrixXd & z);
  MatrixXd linear(const MatrixXd & z);
  MatrixXd linear_grad(const MatrixXd & z);
  MatrixXd sigmoid(const MatrixXd & z);
  MatrixXd sigmoid_grad(const MatrixXd & z);
  
  double cost(const MatrixXd & pred, const MatrixXd & y);
  MatrixXd cost_grad(const MatrixXd & pred, const MatrixXd & y);
  double mean_squared_error(const MatrixXd & pred, const MatrixXd & y);
  MatrixXd mse_grad(const MatrixXd & pred, const MatrixXd & y);
  double cross_entropy(const MatrixXd & pred, const MatrixXd & y);
  MatrixXd ce_grad(const MatrixXd & pred, const MatrixXd & y);
  
  
  // techniques
  void lr_update(double & _alpha, const int & _iter);
  void lr_epoch_thres(double & _alpha, const int & _iter);
  void lr_scaling(double & _alpha, const int & _iter);
  void lr_halve(double & _alpha, const int & _iter);
  void lr_step(double & _alpha, const int & _iter);
  void momentum_update(double & _mmt_param, const int & _iter);
  double reg_cost(const MatrixXd & pred, const MatrixXd & y);
  double reg_L1(const MatrixXd & pred, const MatrixXd & y);
  double reg_L2(const MatrixXd & pred, const MatrixXd & y);
  void reg_grad(MxPtr & _reg_grad, const int & _num_rg);
  void reg_no_grad(MxPtr & _reg_grad, const int & _num_rg);
  void reg_L1_grad(MxPtr & _reg_grad, const int & _num_rg);
  void reg_L2_grad(MxPtr & _reg_grad, const int & _num_rg);
  
  
 public:
  
  ANN();
  ANN(const VectorXi & _layer_size, const VectorXi & _activ_func, 
      const CostFunction & _cost_func);
  ~ANN();
  
  // train
  void train(const MatrixXd & x, const MatrixXd & y, const int & _type);
  void train_val(const MatrixXd & x, const MatrixXd & y,
		 const MatrixXd & xval, const MatrixXd & yval, const int & _type);
  
  // predict and evaluation functions
  MatrixXd predict(const MatrixXd & x, const int & _type);
  double evaluate_mse(const MatrixXd & x, const MatrixXd & y);
  void evaluate_classify(const MatrixXd & x, const MatrixXd & y, MxPtr & eval);
  
  
  // setup functions
  void set_arch(const VectorXi & _layer_size, const VectorXi & _activ_func, 
		const CostFunction & _cost_func);
  void set_weights(const WeightInit & _type);
  void set_grad_desc(const int & _max_epoch, const GradDesc & _type);
  void set_grad_desc(const int & _max_epoch, const GradDesc & _type, 
		     const int & _batch_size);
  void set_lr(const double & _alpha);
  void set_lr(const double & _alpha, const LRDecay & _type, 
	      const int & _epoch_thres, const EpochIter & _ei);
  void set_momentum(const Moment & _type);
  void set_momentum(const Moment & _type, const double & _init, 
		    const double & _final, const int & _epoch_low, 
		    const int & _epoch_upp, const EpochIter & _ei);
  
  void set_regulariz(const Regularize & _type);
  void set_regulariz(const Regularize & _type, const double & _lambda);
  
  void set_early_stop(EarlyStop _type, const int & _stop_thres);
  
  // parameter retrieve functions
  MxPtr get_init_weights();
  MxPtr get_train_weights();
  vector<double>* get_train_cost();
  vector<double>* get_val_cost();
  vector<double>* get_train_ce();
  vector<double>* get_val_ce();
};

#endif
