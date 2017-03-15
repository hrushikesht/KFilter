
//' Function For Predicting States using Kalman Filter
//' 
//' The function allows you input the parameter matrices of
//' the Kalman Filter and returns a matrix containing
//' predicted states/mesurements using given parameters.
//'
//' @param z    Best estimate of measurements
//' @param x    Initial state
//' @param f    Prediction matrix used to obtain next state
//' @param h    Matrix to model the signal
//' @param q    Covariance matrix to model uncertainity associated with untracked influences 
//' @param r    Covariance matrix to model uncertainity eg. sensor noise
//' @param p    Error covariance matrix of the state estimate 
//' @keywords   KalmanFilter
//' 
//' @return Updated states/measurements


#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::export]]
arma::mat kalmanFilter(NumericMatrix z,NumericMatrix x,
        NumericMatrix f,NumericMatrix p,
        NumericMatrix h,NumericMatrix q,NumericMatrix r){
  
    //typecasting
    arma::mat Z = as<arma::mat>(z);
    arma::mat X = as<arma::mat>(x);
    arma::mat F = as<arma::mat>(f);
    arma::mat P = as<arma::mat>(p);
    arma::mat H = as<arma::mat>(h);
    arma::mat Q = as<arma::mat>(q);
    arma::mat R = as<arma::mat>(r);


    //initialising predicted states
    int rows=Z.n_rows;
    int cols=Z.n_cols;

    arma::mat res(Z.n_cols,X.n_rows);

    arma::mat Xb;

    for(int i=0;i<rows;++i){

        //Predict Step
        Xb = F*X;
        P = F*P*F.t() + Q;

        //Update Step
        arma::mat Y = Z.row(i) - (H*Xb);
        arma::mat S = (H*P*H.t())+R;
        arma::mat K = P*H.t()*inv(S);

        Xb = Xb + K*Y;
        P = P - (K*H*P);
        Xb = X;

        res.row(i)=Xb.t();

    }

    return(res);
}

