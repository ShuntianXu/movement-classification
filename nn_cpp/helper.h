#ifndef HELPER_H
#define HELPER_H

#include "Eigen/Dense"
#include <vector>

using namespace Eigen;

// load data from the text file and save data and label into x and y
void load_data(const char* filename, MatrixXd & x, MatrixXd & y);

// save the cost or classification error into a text file
void save_cost(const char* filename, std::vector<double> * cost);

// print the evaluation result for movement classification
void print_eval_classify(MatrixXd * eval);

#endif