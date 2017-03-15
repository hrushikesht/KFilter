// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// kalmanFilter
arma::mat kalmanFilter(NumericMatrix z, NumericMatrix x, NumericMatrix f, NumericMatrix p, NumericMatrix h, NumericMatrix q, NumericMatrix r);
RcppExport SEXP KFilter_kalmanFilter(SEXP zSEXP, SEXP xSEXP, SEXP fSEXP, SEXP pSEXP, SEXP hSEXP, SEXP qSEXP, SEXP rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type z(zSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type f(fSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type p(pSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type h(hSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type q(qSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type r(rSEXP);
    rcpp_result_gen = Rcpp::wrap(kalmanFilter(z, x, f, p, h, q, r));
    return rcpp_result_gen;
END_RCPP
}
