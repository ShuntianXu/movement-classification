#include "ann.h"

ANN::ANN() {
  layer.num = 0;
  layer.size = nullptr;
  layer.af = nullptr;
  layer.cf = CrossEntropy;
  
  weight.init = nullptr;
  weight.train = nullptr;
  set_weights(Uniform);
  
  gd.cost_train = nullptr;
  gd.cost_val = nullptr;
  gd.ce_train = nullptr;
  gd.ce_val = nullptr;
  set_grad_desc(50, Stochastic);
  set_lr(0.1);
  set_momentum(NO_Mmt);
  set_regulariz(NO_Reg);
  set_early_stop(OFF, 0);
}

ANN::ANN(const VectorXi & _layer_size, const VectorXi & _activ_func, 
	 const CostFunction & _cost_func) {
  
  ANN();
  set_arch(_layer_size, _activ_func, _cost_func);
}

ANN::~ANN() {
  delete_arch();
  delete_weights();
  delete_cost();
  
  // remember to delete all new
}

void ANN::assign_new_mx(MxPtr & ptr, const int & num) {
  
  try {
    ptr = new MatrixXd[num];
  } catch(bad_alloc) {
    cerr << "Error: ran out of memory" << endl;
    exit(EXIT_FAILURE);
  }
}

/* ===========================================================================
                                 Architecture
   =========================================================================== */

void ANN::set_arch(const VectorXi & _layer_size, const VectorXi & _activ_func, 
		   const CostFunction & _cost_func) {
  
  if(!check_arch(_layer_size, _activ_func, _cost_func))
    exit(EXIT_FAILURE);
  
  layer.num = _activ_func.size();
  
  layer.size = new int[layer.num+1];
  layer.af = new Activation[layer.num];
  for(int i=0; i<layer.num; i++) {
    layer.size[i] = _layer_size(i);
    layer.af[i] = Activation(_activ_func(i));
  }
  layer.size[layer.num] = _layer_size(layer.num);
  
  layer.cf = _cost_func;
}

bool ANN::check_arch(const VectorXi & _layer_size, 
		     const VectorXi & _activ_func, 
		     const CostFunction & _cost_func) {
  
  if(_layer_size.size()<2) {
    cerr << "Error: Incorrect number of layers\n"
	 << "The number of layers (including input, hidden, "
	 << "and output layers) should be at least two" << endl;
    return false;
  }
  
  if(_layer_size.size() != (_activ_func.size()+1)) {
    cerr << "Error: The number of activation function does not match "
	 << "the number of layers" << endl;
    return false;
  }
  
  for(int i=0; i<_layer_size.size(); i++) {
    if(_layer_size(i)<=0) {
      cerr << "Error: Incorrect layer size\n"
	   << "The layer size should be a positive integer" << endl;
      return false;
    }
  }
  
  for(int i=0; i<_activ_func.size(); i++) {
    if(_activ_func(i)!=Linear // && _activ_func(i)!=ReLU
       &&_activ_func(i)!=Sigmoid /*&& _activ_func(i)!=Softmax*/) {
      cerr << "Error: Incorrect type of activation function" << endl;
      return false;
    }
  }
  
  if(_cost_func!=MeanSquaredError && _cost_func!=CrossEntropy) {
    cerr << "Error: Incorrect type of cost function" << endl;
    return false;
  }
  
  delete_arch();
  
  return true;
}

void ANN::delete_arch() {
  if(layer.size!=nullptr) {
    delete [] layer.size;
    layer.size = nullptr;
  }
  if(layer.af!=nullptr) {
    delete [] layer.af;
    layer.af = nullptr;
  }
}


/* ===========================================================================
                                    Weight
   =========================================================================== */

void ANN::set_weights(const WeightInit & _type) {
  
  if(!check_weights(_type))
    exit(EXIT_FAILURE);
  
  weight.type = _type;
  delete_weights();
  init_weights();
}

bool ANN::check_weights(const WeightInit & _type) {
  
  if(_type!=Uniform && _type!=Glorot && _type!=He) {
    cerr << "Error: Incorrect weights initialization type" << endl;
    return false;
  }
  
  return true;
}

void ANN::delete_weights() {
  
  if(weight.init!=nullptr) {
    delete [] weight.init;
    weight.init = nullptr;
  }
  if(weight.train!=nullptr) {
    delete [] weight.train;
    weight.train = nullptr;
  }
}

void ANN::init_weights() {
  
  assign_new_mx(weight.init, layer.num);
  assign_new_mx(weight.train, layer.num);
  
  if(weight.type==Uniform)
    init_weights_uniform();
  else
    init_weights_gaussian();
}

void ANN::init_weights_uniform() {
  
  std::srand(unsigned(std::time(0)));
  for(int i=0; i<layer.num; i++) {
    double epsilon_init = sqrt(2.0/(layer.size[i]+layer.size[i+1]));
    MatrixXd temp = MatrixXd::Random(layer.size[i+1], layer.size[i]+1);
    weight.init[i] = temp.array() * epsilon_init;
    weight.train[i] = weight.init[i];
  }
}

void ANN::init_weights_gaussian() {
  
  // default_random_engine generator;
  std::random_device rd;
  std::mt19937 generator(rd());
  double sigma;
  
  for(int i=0; i<layer.num; i++) {
    
    sigma = (weight.type==He) ? 
      sqrt(2.0/layer.size[i]) : sqrt(2.0/(layer.size[i]+layer.size[i+1]));
    
    normal_distribution<double> distribution(0.0, sigma);
    
    weight.init[i] = MatrixXd::Zero(layer.size[i+1], layer.size[i]+1);
    for(int r=0; r<layer.size[i+1]; r++) {
      for(int c=0; c<layer.size[i]+1; c++)
	weight.init[i](r,c) = distribution(generator);
    }
    
    weight.train[i] = weight.init[i];
  }
}

MxPtr ANN::get_init_weights() {
  return weight.init;
}

MxPtr ANN::get_train_weights() {
  return weight.train;
}


/* ===========================================================================
                               Gradient Descent
   =========================================================================== */

void ANN::set_grad_desc(const int & _max_epoch, const GradDesc & _type) {
  
  if(!check_grad_desc(_max_epoch, _type))
    exit(EXIT_FAILURE);
  
  set_grad_desc(_max_epoch, _type, 1);
}

void ANN::set_grad_desc(const int & _max_epoch, 
			const GradDesc & _type, 
			const int & _batch_size) {
  
  if(!check_grad_desc(_max_epoch, _type, _batch_size))
    exit(EXIT_FAILURE);
  
  delete_cost();
  
  gd.max_epoch = _max_epoch;
  gd.type = _type;
  gd.batch_size = _batch_size;
  if(_type==Stochastic)
    gd.batch_size = 1;
  gd.cost_train = new vector<double>;
  gd.cost_val = new vector<double>;
  gd.ce_train = new vector<double>;
  gd.ce_val = new vector<double>;
}

bool ANN::check_grad_desc(const int & _max_epoch, 
			  const GradDesc & _type) {
  
  if(_type==MiniBatch) {
    cerr << "Error: Missing batch size for mini batch "
	 << "gradient descent" << endl;
    return false;
  }
  
  return true;
}

bool ANN::check_grad_desc(const int & _max_epoch, 
			  const GradDesc & _type,  
			  const int & _batch_size) {
  
  if(_max_epoch<=0) {
    cerr << "Error: The number of max epoch should be "
	 << "a positive integer"<< endl;
    return false;
  }
  
  if(_type!=Stochastic && _type!=Batch && _type!=MiniBatch) {
    cerr << "Error: Incorrect gradient descent type"<< endl;
    return false;
  }
  
  if(_type==MiniBatch && _batch_size<1) {
    cerr << "Error: The batch size for mini batch gradient descent "
	 << "should be a positive integer" << endl;
    return false;
  }
  
  return true;
}

void ANN::delete_cost() {
  if(gd.cost_train!=nullptr)
    delete gd.cost_train;
  if(gd.cost_val!=nullptr)
    delete gd.cost_val;
  if(gd.ce_train!=nullptr)
    delete gd.ce_train;
  if(gd.ce_val!=nullptr)
    delete gd.ce_val;
}

vector<double>* ANN::get_train_cost() {
  return gd.cost_train;
}

vector<double>* ANN::get_val_cost() {
  return gd.cost_val;
}

vector<double>* ANN::get_train_ce() {
  return gd.ce_train;
}

vector<double>* ANN::get_val_ce() {
  return gd.ce_val;
}


/* ===========================================================================
                                Learning Rate
   =========================================================================== */

void ANN::set_lr(const double & _alpha) {
  set_lr(_alpha, NO_Decay, 0, Epoch);
}

void ANN::set_lr(const double & _alpha, const LRDecay & _type, 
		 const int & _epoch_thres, const EpochIter & _ei) {
  
  if(!check_lr(_alpha, _type, _epoch_thres, _ei))
    exit(EXIT_FAILURE);
  
  lr.alpha = _alpha;
  lr.type = _type;
  lr.epoch_thres = _epoch_thres;
  lr.ei = _ei;
}

bool ANN::check_lr(const double & _alpha, const LRDecay & _type, 
		   const int & _epoch_thres, const EpochIter & _ei) {
  
  if(_alpha<=0) {
    cerr << "Error: Alpha should be a positive number" << endl;
    return false;
  }
  
  if(_type!=NO_Decay && _type!=EpochThres && _type!=Scal
     && _type!=Halve && _type!=Step) {
    cerr << "Error: Incorrect learning rate decay type" << endl;
    return false;
  }
  
  if(_ei!=Epoch && _ei!=Iter) {
    cerr << "Error: Incorrect epoch/iter typ for learning rate decay\n"
	 << "Please specify the type of epoch/iter threshold " 
	 << "using valid keyword: Epoch/Iter" << endl;
    return false;
  }
  
  if(_type!=NO_Decay && _epoch_thres<=0) {
    cerr << "Error: The epoch threshold should be a positive "
	 << "Integer" << endl;
    return false;
  }
  
  return true;
}


void ANN::lr_update(double & _alpha, const int & _iter) {
  
  switch(lr.type) {
  case EpochThres: lr_epoch_thres(_alpha, _iter); break;
  case Scal:       lr_scaling(_alpha, _iter); break;
  case Halve:      lr_halve(_alpha, _iter); break;
  case Step:       lr_step(_alpha, _iter); break;
  default:         return; // NO_Decay
  }
}

void ANN::lr_epoch_thres(double & _alpha, const int & _iter) {
  if(_iter > lr.epoch_thres)
    _alpha = lr.alpha * lr.epoch_thres / _iter;
}

void ANN::lr_scaling(double & _alpha, const int & _iter) {
  if(_iter > lr.epoch_thres)
    _alpha *= 0.99;
}

void ANN::lr_halve(double & _alpha, const int & _iter) {
  _alpha = lr.alpha / pow(2, _iter/lr.epoch_thres);
}

void ANN::lr_step(double & _alpha, const int & _iter) {
  _alpha = lr.alpha / (1 + _iter/lr.epoch_thres);
}


/* ===========================================================================
                                   Momentum
   =========================================================================== */

void ANN::set_momentum(const Moment & _type) {
  
  if(!check_momentum(_type))
    exit(EXIT_FAILURE);
  
  set_momentum(_type, 0, 0, 0, 0, Epoch);
}

void ANN::set_momentum(const Moment & _type, const double & _init, 
		       const double & _final, const int & _epoch_low, 
		       const int & _epoch_upp, const EpochIter & _ei) {
  
  if(!check_momentum(_type, _init, _final, _epoch_low, _epoch_upp, _ei))
    exit(EXIT_FAILURE);
  
  mmt.type = _type;
  mmt.init = _init;
  mmt.final = _final;
  mmt.epoch_low = _epoch_low;
  mmt.epoch_upp = _epoch_upp;
  mmt.ei = _ei;
}

bool ANN::check_momentum(const Moment & _type) {
  
  if(_type!=NO_Mmt) {
    cerr << "Error: Missing momentum specification" << endl;
    return false;
  }
  
  return true;
}

bool ANN::check_momentum(const Moment & _type, const double & _init, 
			 const double & _final, const int & _epoch_low, 
			 const int & _epoch_upp, const EpochIter & _ei) {
  
  if(_type!=NO_Mmt && _type!=LinInc) {
    cerr << "Error: Incorrect momentum type" << endl;
    return false;
  }
  
  if(_type!=NO_Mmt && _init<0) {
    cerr << "Error: Initial momentum should be zero "
	 << "or a positive number" << endl;
    return false;
  }
  
  if(_type!=NO_Mmt && _final<_init) {
    cerr << "Error: Final momentum should be equal to "
	 << "or greater than initial momentum" << endl;
    return false;
  }
  
  if(_ei!=Epoch && _ei!=Iter) {
    cerr << "Error: Incorrect epoch/iter type for momentum\n"
	 << "Please specify the type of lower and upper epoch/iter " 
	 << "thresholds using valid keyword: Epoch/Iter" << endl;
    return false;
  }
  
  if(_type!=NO_Mmt && _epoch_low<1) {
    cerr << "Error: Lower epoch threshold should be "
	 << "a positive integer" << endl;
    return false;
  }
  
  if(_type!=NO_Mmt && _epoch_upp<_epoch_low) {
    cerr << "Error: Upper epoch threshold should be equal to "
	 << "or greater than lower epoch threshold" << endl;
    return false;
  }
  
  return true;
}

void ANN::momentum_update(double & _mmt_param, const int & _iter) {
  
  if(mmt.type==LinInc) {
    if(_iter==1)
      _mmt_param = 0;
    else if(_iter<=mmt.epoch_low)
      _mmt_param = mmt.init;
    else if(_iter>mmt.epoch_low && _iter<mmt.epoch_upp)
      _mmt_param += (mmt.final-mmt.init)/(mmt.epoch_upp-mmt.epoch_low);
    else if(_iter>=mmt.epoch_upp)
      _mmt_param = mmt.final;
  }
  else
    return;
}


/* ===========================================================================
                                Regularization
   =========================================================================== */

void ANN::set_regulariz(const Regularize & _type) {
  
  if(!check_regulariz(_type))
    exit(EXIT_FAILURE);

  set_regulariz(_type, 0);
}

void ANN::set_regulariz(const Regularize & _type,
			const double & _lambda) {
  
  if(!check_regulariz(_type, _lambda))
    exit(EXIT_FAILURE);
  
  reg.type = _type;
  reg.lambda = _lambda;
}

bool ANN::check_regulariz(const Regularize & _type) {
  
  if(_type!=NO_Reg) {
    cerr << "Error: Missing regularization parameter" << endl;
    return false;
	}
  
  return true;
}

bool ANN::check_regulariz(const Regularize & _type,
			  const double & _lambda) {
  
  if(_type!=NO_Reg && _type!=L1 && _type!=L2) {
    cerr << "Error: Incorrect regularization type" << endl;
    return false;
  }
  
  if(_type!=NO_Reg && _lambda<0) {
    cerr << "Error: regularization parameter should be zero or "
	 << "a positive number" << endl;
    return false;
  }
  
  return true;
}

double ANN::reg_cost(const MatrixXd & pred, const MatrixXd & y) {
  
  switch(reg.type) {
  case L1:     return reg_L1(pred, y);
  case L2:     return reg_L2(pred, y);
  default:     return 0; // NO_Reg
  }
}

double ANN::reg_L1(const MatrixXd & pred, const MatrixXd & y) {
  
  double reg_term = 0;
  for(int i=0; i<layer.num ; i++) {
    reg_term += weight.train[i].rightCols(layer.size[i+1])
      .array().pow(2).sum();
  }
  reg_term *= reg.lambda/(2.0*gd.batch_size);
  
  return reg_term;
}

double ANN::reg_L2(const MatrixXd & pred, const MatrixXd & y) {
  
  double reg_term = 0;
  for(int i=0; i<layer.num ; i++) {
    reg_term += weight.train[i].rightCols(layer.size[i+1])
      .array().abs().sum();
  }
  reg_term *= reg.lambda/(2.0*gd.batch_size);
  
  return reg_term;
}

void ANN::reg_grad(MxPtr & _reg_grad, const int & _num_rg) {
  
  assign_new_mx(_reg_grad, _num_rg); // _num_rg = layer.num
  
  switch(reg.type) {
  case L1:     reg_L1_grad(_reg_grad, _num_rg); break;
  case L2:     reg_L2_grad(_reg_grad, _num_rg); break;
  default:     reg_no_grad(_reg_grad, _num_rg); break; // NO_Reg
  }
}

void ANN::reg_no_grad(MxPtr & _reg_grad, const int & _num_rg) {
  
  for(int i=0; i<_num_rg; i++) {
    _reg_grad[i] = MatrixXd::Zero(layer.size[i+1],layer.size[i]+1);
  }
}

void ANN::reg_L1_grad(MxPtr & _reg_grad, const int & _num_rg) {
  
  for(int i=0; i<_num_rg; i++) {
    _reg_grad[i] = MatrixXd::Zero(layer.size[i+1],layer.size[i]+1);
    _reg_grad[i].rightCols(layer.size[i]) = 
      reg.lambda/gd.batch_size * 
      weight.train[i].rightCols(layer.size[i]).array().sign();
  }
}

void ANN::reg_L2_grad(MxPtr & _reg_grad, const int & _num_rg) {
  
  for(int i=0; i<_num_rg; i++) {
    _reg_grad[i] = MatrixXd::Zero(layer.size[i+1],layer.size[i]+1);
    _reg_grad[i].rightCols(layer.size[i]) = 
      reg.lambda/gd.batch_size * 
      weight.train[i].rightCols(layer.size[i]).array();
  }
}


/* ===========================================================================
                                Early Stopping
   =========================================================================== */
void ANN::set_early_stop(EarlyStop _type, const int & _stop_thres) {
  
  if(!check_early_stop(_type, _stop_thres))
    exit(EXIT_FAILURE);
  
  estp.type = _type;
  estp.stop_thres = _type==ON ? _stop_thres : 0;
}

bool ANN::check_early_stop(EarlyStop _type, const int & _stop_thres) {
  
  if(_type!=OFF) {
    if(_stop_thres<=0) {
      cerr << "Error: the number of consecutive increases in " 
	   << "cost for early stopping should be a positive integer" 
	   << endl;
      return false;
    }
  }
  
  return true;
}


/* ===========================================================================
                                   Train NN
   =========================================================================== */

void ANN::train(const MatrixXd & x, const MatrixXd & y, const int & _type) {
  MatrixXd xfake = MatrixXd::Zero(1,1), yfake = MatrixXd::Zero(1,1);
  train_nn(x, y, xfake, yfake, false, _type);
}

void ANN::train_val(const MatrixXd & x, const MatrixXd & y,
		    const MatrixXd & xval, const MatrixXd & yval,
		    const int & _type) {
  train_nn(x, y, xval, yval, true, _type);
}

void ANN::train_nn(const MatrixXd & x, const MatrixXd & y,
		   const MatrixXd & xval, const MatrixXd & yval,
		   const bool & val, const int & _type) {
  
  
  if(!check_dataset(x,y,_type))
    exit(EXIT_FAILURE);
  
  if(val && !check_dataset(xval,yval,_type))
    exit(EXIT_FAILURE);
  
  // recheck and adjust parameters according to training set and changes
  reset_params(x);
  
  // map y to vector if classification
  MatrixXd y_vec, yval_vec;
  if(_type==1) {
    y_vec = map_vectors(y); 
    if(val)
      yval_vec = map_vectors(yval);
  }
  else{
    y_vec = y;
    if(val)
      yval_vec = yval;
  }
  
  // feedforward and backpropagation variables
  // net and activation of each nodes in feedforward and weight update
  // save previous update weights if momentum
  MxPtr az = nullptr, weight_grad = nullptr, last_wg = nullptr;  
  int num_az = layer.num * 2 + 1, num_wg = layer.num;
  
  // technical parameters
  double lr_alpha = lr.alpha, mmt_param = 0;
  int lr_iter = 0, mmt_iter = 0;
  
  // early stopping
  int iter = 0;
  int count_ups = 0;
  bool stop = false;
  
  // training subset variables
  MatrixXd x_sub, y_sub, y_sub_ce; // training subset for each iter
  int num_sub = ceil(float(x.rows())/gd.batch_size);
  
  // set random indexes of examples to randomly shuffle training set
  int* eg = nullptr;
  index_rand_shuffle(eg, x.rows());
  
  MxPtr x_mini = nullptr, y_mini = nullptr, y_mini_ce = nullptr;
  if(gd.type==MiniBatch)
    split_minibatch(x_mini, y_mini, y_mini_ce, num_sub, x, y_vec, y, eg);
  
  
  // start training
  for(int i=1; i<=gd.max_epoch; i++) {
    
    for(int n=0; n<num_sub; n++) {
      
      iter += 1; // (i-1)*num_sub+n+1
      
      if(gd.type==Stochastic) {
	x_sub = x.row(eg[n]); y_sub = y_vec.row(eg[n]);
	y_sub_ce = y.row(eg[n]);
      }
      else if(gd.type==Batch) {
	x_sub = x; y_sub = y_vec; y_sub_ce = y;
      }
      else {
	x_sub = x_mini[n]; y_sub = y_mini[n];
	y_sub_ce = y_mini_ce[n];
      }
      
      // learning rate decay and update momentum
      lr_iter = lr.ei==Epoch ? i : iter;
      mmt_iter = mmt.ei==Epoch ? i : iter;
      if(lr.type!=NO_Decay &&
	 ((lr.ei==Epoch && n==0) || lr.ei==Iter))
	lr_update(lr_alpha, lr_iter);
      if(mmt.type!=NO_Mmt &&
	 ((mmt.ei==Epoch && n==0) || mmt.ei==Iter))
	momentum_update(mmt_param, mmt_iter);
      
      // compute gradient
      feedforward(az, num_az, x_sub);
      backpropagate(weight_grad, num_wg, az, num_az, y_sub);
      
      // record cost and ce for train and validation
      if(_type==1) {
	MxPtr eval = nullptr;
	evaluate_classify(x_sub, y_sub_ce, eval);
	double ce = 100 - eval[4](0,0);
	gd.ce_train->push_back(ce);
	delete [] eval; eval = nullptr;
	
	if(val) {
	  evaluate_classify(xval, yval, eval);
	  ce = 100 - eval[4](0,0);
	  gd.ce_val->push_back(ce);
	  delete [] eval; eval = nullptr;
	}
      }
      
      gd.cost_train->push_back(cost(az[layer.num*2],y_sub.transpose()));
      std::cout << "Iteration  " << iter
		<< " | Cost: " << (*gd.cost_train)[iter-1] << endl;
      if(val) {
	MxPtr az_val = nullptr;
	feedforward(az_val, num_az, xval);
	gd.cost_val->push_back(cost(az_val[layer.num*2],yval_vec.transpose()));
	delete [] az_val;
      }
      
      // early stopping
      if(estp.type==ON) {
	if((*gd.cost_train)[iter-1] > (*gd.cost_train)[iter-2]) {
	  count_ups += 1;
	  if(count_ups >= estp.stop_thres) {
	    stop = true; break;
	  }
	}
	else {
	  count_ups = 0;
	}
      }
      
      // first round to record previous change in weight gradient
      if(mmt.type!=NO_Mmt && i==1 && n==0)
	last_wg = weight_grad;
      
      // change of weight
      // mmt_param = 0 if no momentum or momentum but i == 1
      for(int j=0; j<layer.num; j++) {
	weight_grad[j] = 
	  (1-mmt_param)*((-1*lr_alpha)*weight_grad[j].array());
      }
      
      // add momentum if momentum
      if(mmt.type!=NO_Mmt) {
	for(int j=0; j<layer.num; j++) {
	  weight_grad[j] = weight_grad[j].array() + 
	    mmt_param*last_wg[j].array();
	}
      }
      
      // gradient descent
      for(int j=0; j<layer.num; j++)
	weight.train[j] = weight.train[j] + weight_grad[j];
      
      // save the change in weight into last_wg and clear pointers
      if(mmt.type!=NO_Mmt) {
	if(i!=1 || n!=0) {
	  delete [] last_wg; last_wg = nullptr;
	}
	
	if(i!=gd.max_epoch || n!=num_sub-1)
	  // save weight grad in last for momentum update
	  last_wg = weight_grad;
	else
	  delete [] weight_grad;
      }
      else {
	delete [] weight_grad;
      }
      weight_grad = nullptr;
      delete [] az; az = nullptr;
    }
    
    if(estp.type==ON && stop) {
      delete [] last_wg; delete [] weight_grad;
      break;
    }
  }
  
  delete [] eg; eg = nullptr;
  if(gd.type==MiniBatch) {
    delete [] x_mini; x_mini = nullptr;
    delete [] y_mini; y_mini = nullptr;
  }
}


/* ===========================================================================
                                 NN Auxiliary
   =========================================================================== */

MatrixXd ANN::map_labels(const MatrixXd & result) {
  
  MatrixXd y = MatrixXd::Zero(result.rows(), 1);
  std::ptrdiff_t label;
  
  for(int i=0; i<result.rows(); i++) {
    result.row(i).maxCoeff(&label);
    y(i,0) = label + 1;
  }
  
  return y;
}

MatrixXd ANN::map_vectors(const MatrixXd & y) {
  
  MatrixXd y_v = MatrixXd::Zero(y.rows(),layer.size[layer.num]);
  for(int i=0; i<y.rows(); i++) {
    y_v(i,y(i)-1) = 1;
  }
  
  return y_v;
}

void ANN::index_rand_shuffle(int* & _ind, const int & _num) {
  
  try {
    _ind = new int[_num];
  } catch(bad_alloc) {
    cerr << "Error: ran out of memory" << endl;
    exit(EXIT_FAILURE);
  }
  
  std::srand(unsigned(std::time(0)));
  for(int i=0; i<_num; i++)
    _ind[i] = i;
  
  std::random_shuffle(_ind, _ind+_num, rand_aux);
}

void ANN::split_minibatch(MxPtr & _x, MxPtr & _y, MxPtr & _y_ce, const int & _num_sub, 
			  const MatrixXd & x, const MatrixXd & y, const MatrixXd & y_ce,
			  int* _ind) {
  
  assign_new_mx(_x, _num_sub);
  assign_new_mx(_y, _num_sub);
  assign_new_mx(_y_ce, _num_sub);
  
  // initialize subsets
  for(int i=0; i<_num_sub-1; i++) {
    _x[i] = MatrixXd::Zero(gd.batch_size, x.cols());
    _y[i] = MatrixXd::Zero(gd.batch_size, y.cols());
    _y_ce[i] = MatrixXd::Zero(gd.batch_size, y_ce.cols());
  }
  
  int num_last_sub = x.rows()-(_num_sub-1)*gd.batch_size;
  _x[_num_sub-1] = MatrixXd::Zero(num_last_sub, x.cols());
  _y[_num_sub-1] = MatrixXd::Zero(num_last_sub, y.cols());
  _y_ce[_num_sub-1] = MatrixXd::Zero(num_last_sub, y_ce.cols());
  
  // split subsets
  for(int i=0; i<_num_sub-1; i++) {
    for(int j=0; j<gd.batch_size; j++) {
      _x[i].row(j) = x.row(_ind[i*gd.batch_size+j]);
      _y[i].row(j) = y.row(_ind[i*gd.batch_size+j]);
      _y_ce[i].row(j) = y_ce.row(_ind[i*gd.batch_size+j]);
    }
  }
  // last subset if number of examples in last subset less than batch size
  for(int i=0; i<num_last_sub; i++) {
    _x[_num_sub-1].row(i) = x.row(_ind[(_num_sub-1)*gd.batch_size+i]);
    _y[_num_sub-1].row(i) = y.row(_ind[(_num_sub-1)*gd.batch_size+i]);
    _y_ce[_num_sub-1].row(i) = y_ce.row(_ind[(_num_sub-1)*gd.batch_size+i]);
  }
}

bool ANN::check_dataset(const MatrixXd & x, const MatrixXd & y, 
			const int & _type) {
  
  if(_type!=0 && _type!=1) {
    cerr << "Error: Incorrect type, type should be 0 or 1\n"
	 << "0 - regression | 1 - classification" << endl;
    return false;
  }
  
  if(x.cols()!=layer.size[0]) {
    cerr << "Error: The number of features of examples should be "
	 << "equal to the number of input nodes" << endl;
    return false;
  }
  
  if(!_type && y.cols()!=layer.size[layer.num]) {
    cerr << "Error: The number of output per example should be "
	 << "equal to the number of output nodes" << endl;
    return false;
	}
  
  // if(gd.type==MiniBatch && gd.batch_size>x.rows()) {
  // 	cerr << "Error: Batch size for Mini Batch Gradient Desecent " 
  // 		 << "should be less than the number of training examples"
  // 		 << endl;
  // 	return false;
  // }
  
  if(_type==1 && y.cols()!=1) {
    cerr << "Error: The label for classification should be " 
	 << "a vector" << endl;
    return false;
  }
  
  // if(_type && (y.maxCoeff()-y.minCoeff()+1)!=layer.size[layer.num]) {
  // 	cerr << "Error: The number of labels should be "
  // 		 << "equal to the number of output nodes" << endl;
  // 	return false;
  // }
  
  return true;
}

void ANN::reset_params(const MatrixXd & x) {
  if(gd.type==Batch)
    gd.batch_size = x.rows();
  
  set_weights(weight.type); // initialize weights before training
  delete_cost();
  gd.cost_train = new vector<double>;
  gd.cost_val = new vector<double>;
  gd.ce_train = new vector<double>;
  gd.ce_val = new vector<double>;
}


/* ===========================================================================
                           Feedforward & Backpropagation
   =========================================================================== */

void ANN::feedforward(MxPtr & _az, const int & _num_az, 
		      const MatrixXd & x) {
  
  int m = x.rows();
  
  assign_new_mx(_az, _num_az); // _num_az = layer.num * 2 + 1
  
  _az[0] = MatrixXd::Ones(1+layer.size[0], m);
  _az[0].bottomRows(layer.size[0]) = x.transpose();
  
  for(int i=1; i<_num_az; i+=2) {
    
    _az[i] = weight.train[i/2] * _az[i-1];
    
    // the size of a[i+1] should be the same as _az[i] with one more row
    if(i+1<_num_az-1) {
      _az[i+1] = MatrixXd::Ones(1+layer.size[i/2+1], m);
      _az[i+1].bottomRows(layer.size[i/2+1]) = 
	activate(layer.af[i/2], _az[i]);
    }
    else
      _az[i+1] = activate(layer.af[i/2], _az[i]);
  }
}

void ANN::backpropagate(MxPtr & _weight_grad, const int & _num_wg, 
			const MxPtr & _az, const int & _num_az, 
			const MatrixXd & y) {
  
  assign_new_mx(_weight_grad, _num_wg); // _num_wg = layer.num
  MxPtr delta;
  assign_new_mx(delta, _num_wg);        // _num_wg = layer.num
  
  // regularization
  MxPtr rg_grad = nullptr;
  if(reg.type!=NO_Reg)		
    reg_grad(rg_grad, _num_wg);
  
  // _num_az = layer.num * 2 + 1
  delta[_num_wg-1] = (cost_grad(_az[_num_az-1], y.transpose())).array()
    * (activ_grad(layer.af[_num_wg-1], _az[_num_az-2])).array();
  
  for(int i=_num_wg-2; i>=0; i--) {
    
    delta[i] = (weight.train[i+1].rightCols(layer.size[i+1]).transpose() 
		* delta[i+1]).array() * activ_grad(layer.af[i], _az[2*i+1]).array();
  }
  
  for(int i=_num_wg-1; i>=0; i--) {
    
    _weight_grad[i] = (delta[i] * _az[2*i].transpose()).array() / 
      gd.batch_size;
    
    if(reg.type!=NO_Reg)
      _weight_grad[i] += rg_grad[i];
  }
  
  delete [] delta; delta = nullptr;
  if(reg.type!=NO_Reg) {
    delete [] rg_grad; rg_grad = nullptr;
  }
}

/* =========================================================================== 
                              Activation Function
   =========================================================================== */

MatrixXd ANN::activate(const Activation & type, const MatrixXd & z) {
  
  switch(type) {
  case Linear:  return linear(z);
  case Sigmoid: return sigmoid(z);
    // case ReLU:    return rectifier(z);
    // case Softmax: return softmax(z);
  default:      return linear(z);
  }
}

MatrixXd ANN::activ_grad(const Activation & type, const MatrixXd & z) {
  
  switch(type) {
  case Linear:  return linear_grad(z);
  case Sigmoid: return sigmoid_grad(z);
    // case ReLU:    return rectifier_grad(z);
    // case Softmax: return softmax_grad(z);
  default:      return linear_grad(z);
  }
}

MatrixXd ANN::linear(const MatrixXd & z) {
  return z;
}

MatrixXd ANN::linear_grad(const MatrixXd & z) {
  return MatrixXd::Ones(z.rows(), z.cols());
}

MatrixXd ANN::sigmoid(const MatrixXd & z) {
  return inverse(1.0 + exp(-1 * z.array()));
}

MatrixXd ANN::sigmoid_grad(const MatrixXd & z) {
  return (sigmoid(z).array()) * (1 - sigmoid(z).array());
}


/* =========================================================================== 
                                Cost Function
   =========================================================================== */

double ANN::cost(const MatrixXd & pred, const MatrixXd & y) {
  
  switch(layer.cf) {
  case MeanSquaredError: return mean_squared_error(pred, y);
  case CrossEntropy:     return cross_entropy(pred, y);
    // case NegLogLike:    return neg_log_like(pred, y);
  default:               return mean_squared_error(pred, y);
  }
}

MatrixXd ANN::cost_grad(const MatrixXd & pred, const MatrixXd & y) {
  
  switch(layer.cf) {
  case MeanSquaredError: return mse_grad(pred, y);
  case CrossEntropy:     return ce_grad(pred, y);
    // case NegLogLike:    return nll_grad(pred, y);
  default:               return mse_grad(pred, y);
  }
}

double ANN::cross_entropy(const MatrixXd & pred, const MatrixXd & y) {
  return (-1.0/gd.batch_size * (y.array() * log(pred.array()) + 
				(1.0-y.array()) * log(1-pred.array())).sum());
}

double ANN::mean_squared_error(const MatrixXd & pred, const MatrixXd & y) {
  return 1.0/(2*gd.batch_size) * (pred - y).array().pow(2).sum();
}

MatrixXd ANN::mse_grad(const MatrixXd & pred, const MatrixXd & y) {
  return (pred - y);
}

MatrixXd ANN::ce_grad(const MatrixXd & pred, const MatrixXd & y) {
  return ((pred-y).array()/(pred.array()*(1-pred.array())));
}


/* ===========================================================================
                             Prediction & Evaluation
   =========================================================================== */

double ANN::evaluate_mse(const MatrixXd & x, const MatrixXd & y) {
  
  if(!check_dataset(x,y,0))
    exit(EXIT_FAILURE);
  
  MatrixXd pred = predict(x, 0);
  return mean_squared_error(pred, y);
}

void ANN::evaluate_classify(const MatrixXd & x, const MatrixXd & y,
			    MxPtr & eval) {
  
  if(!check_dataset(x,y,1))
    exit(EXIT_FAILURE);
  if(eval!=nullptr){
    delete [] eval; eval = nullptr;
  }
  
  assign_new_mx(eval, 5);
  MatrixXd pred = predict(x, 1);
  
  // confusion matrix
  int num_labels = layer.size[layer.num];
  eval[0] = MatrixXd::Zero(num_labels, num_labels);
  for(int i=0; i<num_labels; i++) {
    for(int j=0; j<num_labels; j++) {
      for(int a=0; a<y.rows(); a++) {
	if((pred(a)==i+1)&&(y(a)==j+1))
	  eval[0](i,j) += 1;
      }
    }
  }
  
  // precision, recall, F1
  eval[1] = MatrixXd::Zero(num_labels,1);
  eval[2] = MatrixXd::Zero(num_labels,1);
  eval[3] = MatrixXd::Zero(num_labels,1);
  for(int i=0; i<num_labels; i++) {
    eval[1](i) = 100.0 * eval[0](i,i) / eval[0].row(i).sum();
    eval[2](i) = 100.0 * eval[0](i,i) / eval[0].col(i).sum();
    eval[3](i) = 2.0 * eval[1](i)*eval[2](i) / (eval[1](i)+eval[2](i));
  }
  // classification rate
  eval[4] = MatrixXd::Zero(1,1);
  eval[4](0) = 100.0 * eval[0].trace() / eval[0].sum();
}

MatrixXd ANN::predict(const MatrixXd & x, const int & _type) {
  
  if(x.cols()!=layer.size[0]) {
    cerr << "Error: The number of features of examples should be "
	 << "equal to the number of input nodes" << endl;
    exit(EXIT_FAILURE);
  }
  if(_type!=0 && _type!=1) {
    cerr << "Error: Incorrect type, type should be 0 or 1\n"
	 << "0 - regression | 1 - classification" << endl;
    exit(EXIT_FAILURE);
  }
  
  MxPtr az = nullptr; int num_az = 2 * layer.num + 1;
  feedforward(az, num_az, x);
  
  // regression
  if(_type==0)
    return az[num_az-1];
  // classification
  else
    return map_labels(az[num_az-1].transpose());
}
