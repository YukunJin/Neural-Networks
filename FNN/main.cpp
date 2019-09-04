#include <vector>
#include <iostream>
#include<string>
#include <random>
#include <math.h>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
using namespace std;
using namespace Eigen;

class Layer{
  MatrixXd Weights;
  VectorXd bias;
  int layer;
  int n;
  int m;

public:
  Layer(int m,int n,int layer){
    Weights = MatrixXd::Random(m,n);
    bias = VectorXd::Random(n);

  }
  void printWeight(){
    std::cout << Weights << '\n';
  }
  void printBias(){
     std::cout << bias << '\n';
  }
  void evaluate(VectorXd &input,VectorXd &output){
    VectorXd temp = (Weights.transpose() * input);
    temp += bias;
    sigmoid(temp,output);
  }

};
class Fnn{
  vector<Layer> networks;
  int number_of_layers;
  VectorXd input;
  VectorXd output;
public:
  Fnn(int layers){
    number_of_layers = layers;
    networks.size = number_of_layers;
    
  }
  void sigmoid(VectorXd &input,VectorXd &output){
    for (int i = 0;i<input.size();++i){
      output(i) = (1/(1+exp(input(i))));
    }
  }
  void d_sigmoid(VectorXd &input, VectorXd &output){
    for (int i = 0;i<input.size();++i){
      output(i) = (exp(-input(i))/(1+pow(exp(-input(i)),2)));
    }
  }

}
int main(){

  VectorXd input(2); input << 1,2;
  VectorXd output(2);

  Layer L_0(2,2,0);
  L_0.evaluate(input,output);
  L_0.printWeight();
  L_0.printBias();
  std::cout << output << '\n';
  return 0;
}
