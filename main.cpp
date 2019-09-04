#include <vector>
#include <iostream>
#include<string>
#include <random>
#include <math.h>
#include <fstream>
#include <eigen3/Eigen/Dense>
using namespace std;
using namespace Eigen;
class Hopfield{
  vector<int > state;
  vector<int > biases;
  std::vector<std::vector<int> > W;
public:
  void setRandomHopfield(int N){
    setRandomState(N);
    setRandomBiases(N);
    setRandomWeights(N);
  }

  vector<int> getState(){
    return state;
  }
  /**
  **
  **/
  void setRandomState(int N){

    state.resize(N);
    int random;
    for (int i=0;i<N;i++){
        random = rand()%2;
        if(random == 0){
          state[i] = -1;
        }
        else{
          state[i] = 1;
        }
      }
    }

  /**
  **
  **/
  void setRandomBiases(int N){
    biases.resize(N);
    int random;
    for (int i=0;i<N;++i){
      random = rand()%2 ;
      if(random == 0){
        biases[i] = -1;
      }
      else{
        biases[i] = 1;
      }
    }
  }
  /**
  **
  **/
  void setRandomWeights(int N){
    W.resize(N);
    int random;
    for(int i=0;i<N;++i){
      W[i].resize(N);
      W[i][i] = 0;
    }
    for (int i = 0; i<N;++i){
      for(int j=i+1;j<N;++j){
        random = rand()%2;
        if(random == 0){
          W[i][j] = -1;
          W[j][i] = -1;
        }
        else{
          W[i][j] = 1;
          W[j][i] = 1;
        }
      }
    }
  }
  void setWeight(vector<vector<int> > &v){
    for (int i=0;i<v.size();i++){
      for(int j=0;j<v[i].size();j++){
        W[i][j] = v[i][j];
      }
    }
  }
  void setState(vector<int> &v){
    for(int i=0;i<v.size();++i){
      state[i] = v[i];
    }
  }
  void printWeightMat(){
    for (int i = 0;i<W.size();++i){
      for (int j = 0;j<W[i].size();++j){
        std::cout << W[i][j] ;
        cout << ' ';
      }
      std::cout << '\n';
    }
  }
  bool checkConverge(){
    double prev_Eng = calcEng();
    for(int i = 0;i<state.size();++i){
      updateState(i);
      if(calcEng()!=prev_Eng){
        state[i] = -state[i];
        return false;
      }
      else{
        continue;
      }
    }
    return true;

  }
  void updateState(int neuronId){
    int totalWeight = 0;
    for (int i = 0;i<W[neuronId].size();++i){
      totalWeight += W[neuronId][i] * state[i];
    }
    if (totalWeight > biases[neuronId]){
      state[neuronId] = 1;
    }
    else{
      state[neuronId] = -1;
    }
  }
  double calcEng(){
    int weightSum = 0;
    int biasesSum = 0;
    for (int i = 0;i<W.size();++i){
      biasesSum += biases[i] * state[i];
      for (int j = 0;j<W[i].size();++j){
        weightSum += W[i][j]*state[i]*state[j];
      }
    }
    return -0.5 * weightSum + biasesSum;
  }
};
vector<int> shuffleState(VectorXd input){
  std::vector<int> numbers;
  for(int i=0;i<100;i++)
    numbers.push_back(i);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(numbers.begin(), numbers.end(), std::default_random_engine(seed));
  vector<int> ret(input.size());
  for (int i = 0 ; i<ret.size();++i){
    ret[i] = input(i);
  }
  for(int i=0; i<20; i++){
    if(rand()%2 == 0){
      ret[numbers[i]] = -1;
    }
    else{
      ret[numbers[i]] = 1;
    }
  }
  return ret;
}
VectorXd readImgFile(string fileName){
  vector<int> temp;
  fstream file;
  file.open("imgs/"+fileName);
  int state;
  while (file >> state){
      if(state == 0){
    temp.push_back(-1);
      }
      else{
    temp.push_back(1);
      }
    }
  VectorXd v(temp.size());
  for (int i=0;i<temp.size();++i){
    v[i] = temp[i];
    }
  return v;
  }
vector<vector<int > > weighCompute(vector<VectorXd> memories){
  int size = memories[0].size();
  MatrixXd weight(memories[0].size(),memories[0].size());
  for(int i = 0 ; i<memories.size();i++){
    weight += memories[i] * memories[i].transpose();
  }
  weight /= memories.size();
  vector<vector<int > > ret;
  ret.resize(size);
  for (int i = 0 ; i<weight.rows() ; ++i){
    for (int j = 0; j<weight.cols() ; ++j){
      ret[i].resize(weight.cols());
      std::cout << i <<' ' << j << '\n';
      ret[i][j] = weight(i,j);
    }
  }
  return ret;
}
int main(){

  Hopfield h;
  int size = 100;
  h.setRandomHopfield(size);
  /*
  string url = "test_files/";
  for(int i = 0;i<10;++i){
    string filename = to_string(i)+".txt";
    ofstream file;
    file.open(url+filename);
    h.setRandomState(size);
    for(int sweep = 0; sweep<16;sweep++){
      for (int step = 0; step<size;step++){
        h.updateState(rand()%size);
      }
      file << h.calcEng() << '\n';
    }
    file.close();
  }
  */

 vector<VectorXd> memories;
 vector<vector<int > > weight;
 VectorXd face_v = readImgFile("face.txt");
 VectorXd tree_v = readImgFile("tree.txt");
 memories.push_back(face_v);
 memories.push_back(tree_v);
 weight = weighCompute(memories);
 h.setWeight(weight);
 vector<int> init_tree = shuffleState(tree_v);
 h.setState(init_tree);
 string url = "train_result/tree/";
 ofstream file;

 for (int sweep = 0; sweep<20;sweep++){
   file.open(url+to_string(sweep)+".txt");
   for (int step = 0;step<size/3;step++){
     h.updateState(rand()%size);
   }
   std::vector<int> v = h.getState();
   for(int i = 0;i<v.size();i++){
     file << v[i] <<'\n';
   }
   file.close();

 }


  return 0;
}
