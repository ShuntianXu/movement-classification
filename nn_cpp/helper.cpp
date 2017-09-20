#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cassert>

#include "helper.h"

using namespace std;


void load_data(const char* filename, MatrixXd & x, MatrixXd & y) {
  
  cout << "Loading data from file '" << filename << "'...";
  
  ifstream in(filename);
  if(!in)
    cout << "Failed!" << endl;
  assert(in);
  
  int num_r = x.rows(), num_c = x.cols(), i = 0;
  
  while(in && i<num_r) {
    for(int j=0; j<num_c; j++)
      in >> x(i, j);
    in >> y(i, 0);
    i++;
  }
  if(i==num_r) {
    cout << "Success" << endl;
  }
  else
    cout << "Failed" << endl;
  
  in.close();
  
  assert(i==num_r);
}

void save_cost(const char* filename, vector<double> * cost) {
  
  cout << "Saving cost to file '" << filename << "'...";
  ofstream out(filename);
  if(!out)
    cout << "Failed!" << endl;
  assert(out);
  
  for(int i=0; i<cost->size(); i++)
    out << (*cost)[i] << '\n';
  
  out.close();
  cout << "Success" << endl;
}

void print_eval_classify(MatrixXd * eval) {
  
  cout << "\n";
  
  string tag[4];
  tag[0] = "press-up";
  tag[1] = "sit-up";
  tag[2] = "lunge";
  tag[3] = "invalid";
  
  cout << setiosflags(ios::left);
  cout << "Confusion Matrix" << endl;
  cout.width(20); cout << "";
  for(int i=0; i<4; i++) {
    cout.width(20);
    cout << tag[i] + "(actual)";
  }
  cout << "\n";
  
  for(int i=0; i<4; i++) {
    cout.width(20);
    cout << tag[i] + "(predict)";
    for(int j=0; j<4; j++) {
      cout.width(20);
      cout << eval[0](i,j);
    }
    cout << "\n";
  }
  cout << "\n";
  
  
  string metr[4];
  metr[0] = "precision";
  metr[1] = "recall";
  metr[2] = "F1";
  metr[3] = "classification rate";
  
  cout << "Evaluation Metrics" << endl;
  cout.width(20); cout << "";
  for(int i=0; i<4; i++) {
    cout.width(20);
    cout << tag[i];
  }
  cout << "\n";
  
  for(int i=0; i<3; i++) {
    cout.width(20);
    cout << metr[i];
    for(int j=0; j<4; j++) {
      cout.width(20);
      cout << eval[i+1](j);
    }
    cout << "\n";
  }
  cout.width(50);
  cout << metr[3] << eval[4](0) << "\n\n";
}
