#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ChebTools/ChebTools.h"

/*
From numpy:
----------
from numpy.polynomial.chebyshev import Chebyshev
c = Chebyshev([1,2,3,4])
print c.coef
print c.deriv(1).coef
print c.deriv(2).coef
print c.deriv(3).coef
*/
TEST_CASE("Expansion derivatives (3rd order)", "")
{
    Eigen::Vector4d c;
    c << 1, 2, 3, 4;

    auto ce = ChebTools::ChebyshevExpansion(c, -1, 1);
    SECTION("first derivative") {
        Eigen::Vector3d c_expected; c_expected << 14,12,24;
        auto d1 = ce.deriv(1);
        auto d1c = d1.coef();
        auto err = std::abs((c_expected - d1c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("second derivative") {
        Eigen::Vector2d c_expected; c_expected << 12, 96;
        auto d2 = ce.deriv(2);
        auto d2c = d2.coef();
        auto err = std::abs((c_expected - d2c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("third derivative") {
        Eigen::VectorXd c_expected(1); c_expected << 96;
        auto d3 = ce.deriv(3);
        auto d3c = d3.coef();
        auto err = std::abs((c_expected - d3c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
}

/*
From numpy:
----------
from numpy.polynomial.chebyshev import Chebyshev
c = Chebyshev([1,2,3,4,5])
print c.coef
print c.deriv(1).coef
print c.deriv(2).coef
print c.deriv(3).coef
print c.deriv(4).coef
*/
TEST_CASE("Expansion derivatives (4th order)", "")
{
    Eigen::VectorXd c(5);
    c << 1, 2, 3, 4, 5;

    auto ce = ChebTools::ChebyshevExpansion(c, -1, 1);
    SECTION("first derivative") {
        Eigen::Vector4d c_expected; c_expected << 14, 52, 24, 40;
        auto d1c = ce.deriv(1).coef();
        auto err = std::abs((c_expected - d1c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("second derivative") {
        Eigen::Vector3d c_expected; c_expected << 172, 96, 240;
        auto d2c = ce.deriv(2).coef();
        auto err = std::abs((c_expected - d2c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("third derivative") {
        Eigen::Vector2d c_expected; c_expected << 96, 960;
        auto d3c = ce.deriv(3).coef();
        auto err = std::abs((c_expected - d3c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("fourth derivative") {
        Eigen::VectorXd c_expected(1); c_expected << 960;
        auto d4c = ce.deriv(4).coef();
        auto err = std::abs((c_expected - d4c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
}

TEST_CASE("Expansion from single monomial term", "")
{
    // From Mason and Handscomb, Chebyshev Polynomials, p. 23
    auto ce = ChebTools::ChebyshevExpansion::from_powxn(4, -1, 1);
    SECTION("Check coefficients",""){
        Eigen::VectorXd c_expected(5); c_expected << 3.0/8.0, 0, 0.5, 0, 1.0/8.0;
        auto err = std::abs((c_expected - ce.coef()).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("Check calculated value",""){
        auto err = std::abs(pow(3.0, 4.0) - ce.y_Clenshaw(3.0));
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
}

TEST_CASE("Expansion from polynomial", "")
{
    Eigen::VectorXd c_poly(4); c_poly << 0, 1, 2, 3;
    Eigen::VectorXd c_expected(4); c_expected << 1.0, 3.25, 1.0, 0.75;

    // From https ://docs.scipy.org/doc/numpy/reference/generated/numpy.polynomial.chebyshev.poly2cheb.html
    auto ce = ChebTools::ChebyshevExpansion::from_polynomial(c_poly, 0, 10);

    auto err = std::abs((c_expected - ce.coef()).sum());
    CAPTURE(err);
    CHECK(err < 1e-100);
}
/*
From numpy:
----------
from numpy.polynomial.chebyshev import Chebyshev
c1 = Chebyshev([1,2,3,4])
c2 = Chebyshev([0.1,0.2,0.3])
print (c1*c2).coef.tolist()
*/
TEST_CASE("Product of expansions", "")
{
    Eigen::VectorXd c1(4); c1 << 1, 2, 3, 4;
    Eigen::VectorXd c2(3); c2 << 0.1, 0.2, 0.3;
    Eigen::VectorXd c_expected(6); c_expected << 0.7499999999999999, 1.6, 1.2000000000000002, 1.0, 0.85, 0.6;

    auto C1 = ChebTools::ChebyshevExpansion(c1);
    auto C2 = ChebTools::ChebyshevExpansion(c2);

    auto err = std::abs((c_expected - (C1*C2).coef()).sum());
    CAPTURE(err);
    CHECK(err < 1e-14);
}
TEST_CASE("Expansion times x", "")
{
    Eigen::VectorXd c1(7); c1 << 1, 2, 3, 4, 5, 6, 7;
    SECTION("default range"){
        auto x = ChebTools::ChebyshevExpansion::factory(1, [](double x) { return x; }, -1, 1);
        auto C = ChebTools::ChebyshevExpansion(c1, -1, 1);
        auto xCcoeffs = (x*C).coef();
        auto times_x_coeffs = C.times_x().coef();
        auto err = (times_x_coeffs.array() - xCcoeffs.array()).cwiseAbs().sum();
        CAPTURE(xCcoeffs);
        CAPTURE(times_x_coeffs);
        CAPTURE(err);
        CHECK(err < 1e-12);
    }
    SECTION("non-default range") {
        double xmin = -0.3, xmax = 4.4;
        auto x = ChebTools::ChebyshevExpansion::factory(1, [](double x) { return x; }, xmin, xmax);
        auto C = ChebTools::ChebyshevExpansion(c1, xmin, xmax);
        auto xCcoeffs = (x*C).coef();
        auto times_x_coeffs = C.times_x().coef();
        auto err = (times_x_coeffs.array() - xCcoeffs.array()).cwiseAbs().sum();
        CAPTURE(xCcoeffs);
        CAPTURE(times_x_coeffs);
        CAPTURE(err);
        CHECK(err < 1e-12);
    }
    SECTION("default range") {
        auto x61 = ChebTools::ChebyshevExpansion::from_powxn(5,-1,1).times_x();
        auto x62 = ChebTools::ChebyshevExpansion::from_powxn(6,-1, 1);
        auto err = (x61.coef().array() - x62.coef().array()).cwiseAbs().sum();
        CAPTURE(err);
        CHECK(err < 1e-12);
    }
    SECTION("default range; inplace") {
        auto x = ChebTools::ChebyshevExpansion::factory(1, [](double x) { return x; }, -1, 1);
        auto C = ChebTools::ChebyshevExpansion(c1, -1, 1);
        auto xC2 = ChebTools::ChebyshevExpansion(c1, -1, 1);
        xC2.times_x_inplace();
        auto xCcoeffs = (x*C).coef();
        auto xC2_coeffs = xC2.coef();
        auto err = (xC2_coeffs.array() - xCcoeffs.array()).cwiseAbs().sum();
        CAPTURE(xCcoeffs);
        CAPTURE(xC2_coeffs);
        CAPTURE(err);
        CHECK(err < 1e-12);
    }
    SECTION("non-default range; inplace") {
        auto x = ChebTools::ChebyshevExpansion::factory(1, [](double x) { return x; }, -2, 3.4);
        auto C = ChebTools::ChebyshevExpansion(c1, -2, 3.4);
        auto xC2 = ChebTools::ChebyshevExpansion(c1, -2, 3.4);
        xC2.times_x_inplace();
        auto xCcoeffs = (x*C).coef();
        auto xC2_coeffs = xC2.coef();
        auto err = (xC2_coeffs.array() - xCcoeffs.array()).cwiseAbs().sum();
        CAPTURE(xCcoeffs);
        CAPTURE(xC2_coeffs);
        CAPTURE(err);
        CHECK(err < 1e-12);
    }
}
TEST_CASE("Transform y=x^3 by sin(y) to be y=sin(x^3)", "")
{
    auto C1 = ChebTools::ChebyshevExpansion::factory(30, [](double x) { return x*x*x; }, -2, 3.4);
    std::function<Eigen::ArrayXd(const Eigen::ArrayXd &)> _sinf = [](const Eigen::ArrayXd &y){ return y.sin(); };
    auto C2 = C1.apply(_sinf);
    std::cout << C1.coef() << std::endl;
    std::cout << C2.coef() << std::endl;
    double y_expected = sin(0.7*0.7*0.7);
    double y = C2.y(0.7);

    auto err = std::abs((y_expected - y)/y);
    CAPTURE(err);
    CHECK(err < 1e-14);
}

TEST_CASE("Sums of expansions", "")
{
    Eigen::VectorXd c4(4); c4 << 1, 2, 3, 4;
    Eigen::VectorXd c3(3); c3 << 0.1, 0.2, 0.3;
    double xmin = 0.1, xmax = 3.8;

    SECTION("same lengths") {
        Eigen::VectorXd c_expected(4); c_expected << 2,4,6,8;
        auto C1 = ChebTools::ChebyshevExpansion(c4, xmin, xmax);
        auto C2 = ChebTools::ChebyshevExpansion(c4, xmin, xmax);

        auto err = std::abs((c_expected - (C1 + C2).coef()).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("first longer lengths") {
        Eigen::VectorXd c_expected(4); c_expected << 1.1, 2.2, 3.3, 4;
        auto C1 = ChebTools::ChebyshevExpansion(c4, xmin, xmax);
        auto C2 = ChebTools::ChebyshevExpansion(c3, xmin, xmax);

        auto err = std::abs((c_expected - (C1 + C2).coef()).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("second longer length") {
        Eigen::VectorXd c_expected(4); c_expected << 1.1, 2.2, 3.3, 4;
        auto C1 = ChebTools::ChebyshevExpansion(c3, xmin, xmax);
        auto C2 = ChebTools::ChebyshevExpansion(c4, xmin, xmax);

        auto err = std::abs((c_expected - (C1 + C2).coef()).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
}
TEST_CASE("Constant value 1.0", "")
{
    Eigen::VectorXd c(1); c << 1.0;
    Eigen::VectorXd x1(1); x1 << 0.5;
    Eigen::VectorXd x2(2); x2 << 0.5, 0.5;
    auto C = ChebTools::ChebyshevExpansion(c, 0, 10);

    double err = std::abs(C.y_recurrence(0.5) - 1.0);
    CAPTURE(err);
    CHECK(err < 1e-100);

    double err1 = (C.y(x1).array() - 1.0).cwiseAbs().sum();
    CAPTURE(err1);
    CHECK(err1 < 1e-100);

    double err2 = (C.y(x2).array() - 1.0).cwiseAbs().sum();
    CAPTURE(err2);
    CHECK(err2 < 1e-100);

}

TEST_CASE("Constant value y=x", "")
{
    Eigen::VectorXd c(2); c << 0.0, 1.0;
    Eigen::VectorXd x1(1); x1 << 0.5;
    Eigen::VectorXd x2(2); x2 << 0.5, 0.5;

    auto C = ChebTools::ChebyshevExpansion(c, -1, 1);

    SECTION("One element with recurrence", ""){
        double err = std::abs(C.y_recurrence(x1(0)) - x1(0));
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("One element with Clenshaw", "") {
        double err = std::abs(C.y_Clenshaw(x1(0)) - x1(0));
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("One element vector", "") {
        double err = (C.y(x1).array() - x1.array()).cwiseAbs().sum();
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("Two element vector",""){
        double err = (C.y(x2).array() - x2.array()).cwiseAbs().sum();
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
}

TEST_CASE("Constant value y=x with generation from factory", "")
{
    Eigen::VectorXd x1(1); x1 << 0.5;

    SECTION("Standard range", "") {
        auto C = ChebTools::ChebyshevExpansion::factory(2, [](double x){ return x; }, -1, 1);
        double err = (C.y(x1).array() - x1.array()).cwiseAbs().sum();
        CAPTURE(err);
        CHECK(err < 1e-14);
    }
    SECTION("Range(0,10)", "") {
        auto C = ChebTools::ChebyshevExpansion::factory(1, [](double x) { return x; }, 0, 10);
        double err = (C.y(x1).array() - x1.array()).cwiseAbs().sum();
        CAPTURE(err);
        CHECK(err < 1e-14);
    }
}
TEST_CASE("product commutativity","") {
    auto rhoRT = 1e3; // Just a dummy variable
    double deltamin = 1e-12, deltamax = 6;
    auto delta = ChebTools::ChebyshevExpansion::factory(1, [](double x) { return x; }, deltamin, deltamax);
    auto one = ChebTools::ChebyshevExpansion::factory(1, [](double x) { return 1; }, deltamin, deltamax);
    Eigen::VectorXd c(10); c << 0,1,2,3,4,5,6,7,8,9;
    auto c0 = ((ChebTools::ChebyshevExpansion(c, deltamin, deltamax)*delta + one)*(rhoRT*delta)).coef();
    auto c1 = ((rhoRT*delta)*(ChebTools::ChebyshevExpansion(c, deltamin, deltamax)*delta + one)).coef();
    double err = (c0.array() - c1.array()).cwiseAbs().sum();
    CAPTURE(err);
    CHECK(err < 1e-14);
}
TEST_CASE("product commutativity with simple multiplication", "") {
    auto rhoRT = 1e3; // Just a dummy variable
    double deltamin = 1e-12, deltamax = 6;
    auto delta = ChebTools::ChebyshevExpansion::factory(1, [](double x) { return x; }, deltamin, deltamax);
    auto one = ChebTools::ChebyshevExpansion::factory(1, [](double x) { return 1; }, deltamin, deltamax);
    Eigen::VectorXd c(10); c << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
    auto c0 = (ChebTools::ChebyshevExpansion(c, deltamin, deltamax)*rhoRT).coef();
    auto c1 = (rhoRT*ChebTools::ChebyshevExpansion(c, deltamin, deltamax)).coef();
    double err = (c0.array() - c1.array()).cwiseAbs().sum();
    CAPTURE(err);
    CHECK(err < 1e-14);
}


//some corner cases if someone wanted to try and initialize a linear ChebyshevExpansion
TEST_CASE("corner cases with linear ChebyshevExpansion",""){
  double error;
  SECTION("root finding of linear ChebyshevExpansion"){
    Eigen::VectorXd coeffs(2);
    coeffs<<0,1;
    ChebTools::ChebyshevExpansion linCheb = ChebTools::ChebyshevExpansion(coeffs,-1,1);
    double root = linCheb.real_roots(true).at(0);
    error = std::abs(root);
    CAPTURE(error);
    CHECK(error<1e-14);
  }

  SECTION("root finding of linear ChebyshevExpansion test 2"){
    Eigen::VectorXd coeffs(3);
    coeffs<<-1,1,0;
    ChebTools::ChebyshevExpansion linCheb = ChebTools::ChebyshevExpansion(coeffs,-1,1);
    double root = linCheb.real_roots(true).at(0);
    error = std::abs(1-root);
    CAPTURE(error);
    CHECK(error<1e-14);
  }

  SECTION("root finding of linear ChebyshevExpansion test 3"){
    Eigen::VectorXd coeffs(3);
    coeffs<<0,1,0;
    ChebTools::ChebyshevExpansion linCheb = ChebTools::ChebyshevExpansion(coeffs,-1,1);
    double root = linCheb.real_roots(true).at(0);
    error = std::abs(root);
    CAPTURE(error);
    CHECK(error<1e-14);
  }

  SECTION("root finding of linear ChebyshevExpansion test 4"){
    Eigen::VectorXd coeffs(3);
    coeffs<<0,0,0;
    ChebTools::ChebyshevExpansion linCheb = ChebTools::ChebyshevExpansion(coeffs,-1,1);
    int roots = linCheb.real_roots(true).size();
    CAPTURE(roots);
    CHECK(roots==0);
    CHECK(linCheb.coef().size()==3);
  }
}


// functions that should return the 4th degree chebyshev polynomial
// double rhs(double x){ return 0;}
// double secondCoeff(double x){ return 1-std::pow(x,2); }
// double firstCoeff(double x){ return -x;}
// double zerothCoeff(double x){ return 16;}
double rhs(double x){ return 1; }
double zerothCoeff(double x){ return 0;}
double pi_constant(double x){ return std::pow(EIGEN_PI,2); }
double rhs2(double x){ return -std::pow(EIGEN_PI,2)*std::sin(EIGEN_PI*x); }
TEST_CASE("Linear BVP Tests",""){
  double error;
  double tol = 1e-12;
  // Eigen::MatrixXd A(2,2);
  // A << 1/(std::cos(EIGEN_PI/4)-std::cos(3*EIGEN_PI/4)), -1/(std::cos(EIGEN_PI/4)-std::cos(3*EIGEN_PI/4)),
  //     -1/(std::cos(3*EIGEN_PI/4)-std::cos(EIGEN_PI/4)), 1/(std::cos(3*EIGEN_PI/4)-std::cos(EIGEN_PI/4));
  // SECTION("first differentiation matrix check"){
  //   error = (A-ChebTools::DiffMatrixLibrary::norder_diff_matrix(1,1)).norm()
  //   CAPTURE(error);
  //   CHECK(error<tol);
  // }
  // SECTION("second differentiation matrix check"){
  //   error = (A*A-ChebTools::DiffMatrixLibrary::norder_diff_matrix(2,1)).norm()
  //   CAPTURE(error);
  //   CHECK(error<tol);
  // }
  // SECTION("third differentiation matrix check"){
  //   error = (A*A*A-ChebTools::DiffMatrixLibrary::norder_diff_matrix(3,1)).norm()
  //   CAPTURE(error);
  //   CHECK(error<tol);
  // }
  // SECTION("fourth differentiation matrix check"){
  //   error = (A*A*A*A-ChebTools::DiffMatrixLibrary::norder_diff_matrix(4,1)).norm()
  //   CAPTURE(error);
  //   CHECK(error<tol);
  // }
  SECTION("Cheb diff eq test"){
    // std::vector<std::function<double(double)>> coeffs = {zerothCoeff,firstCoeff,secondCoeff};
    std::vector<std::function<double(double)>> coeffs = {zerothCoeff,rhs};
    std::vector<double> left_bc = {0,1,-1};
    std::vector<double> right_bc = {0,1,1};
    ChebTools::ChebyshevExpansion cheb_soln = ChebTools::ChebyshevExpansion::cheb_from_bvp(2, coeffs, rhs, left_bc, right_bc, -1, 1);
    // std::cout << "coeffs" <<cheb_soln.coef()<< '\n';
    Eigen::VectorXd soln = Eigen::VectorXd::Zero(3);
    soln(1) = 1;
    // soln(4) = 1;
    error = (soln-cheb_soln.coef()).norm();
    CAPTURE(error);
    CHECK(error<tol);
  }

  SECTION("Sine on [0,1] test"){
    // std::vector<std::function<double(double)>> coeffs = {zerothCoeff,firstCoeff,secondCoeff};
    std::vector<std::function<double(double)>> coeffs = {zerothCoeff,zerothCoeff,rhs};
    std::vector<double> left_bc = {0,1,0};
    std::vector<double> right_bc = {0,1,0};
    ChebTools::ChebyshevExpansion cheb_soln = ChebTools::ChebyshevExpansion::cheb_from_bvp(32, coeffs, rhs2, left_bc, right_bc, 0, 1);
    Eigen::VectorXd true_soln = (Eigen::ArrayXd::LinSpaced(100,0,1)*EIGEN_PI).sin().matrix();
    // std::cout<<"True_soln"<<std::endl;
    // std::cout<<true_soln<<std::endl;
    Eigen::VectorXd approx_soln = cheb_soln.y(Eigen::ArrayXd::LinSpaced(100,0,1).matrix());
    // std::cout<<"approx_soln"<<std::endl;
    // std::cout<<approx_soln<<std::endl;
    error = (true_soln-approx_soln).norm();
    // std::cout<<"Error: "<<error<<std::endl;
    CAPTURE(error);
    CHECK(error<100*tol);
  }
}
double sin1(double x){ return std::sin(x); }
TEST_CASE("Least squares tests",""){
  double error;
  double tol = 1e-12;
  SECTION("Exact interpolation"){
    Eigen::VectorXd xdata(3);
    xdata<<-1,0,1;
    Eigen::VectorXd ydata(3);
    ydata<<1,-1,1;
    ChebTools::ChebyshevExpansion chebLS = ChebTools::ChebyshevExpansion::cheb_from_leastSquares(2, xdata, ydata);
    Eigen::VectorXd soln = Eigen::VectorXd::Zero(3);
    soln(2) = 1;
    error = (soln-chebLS.coef()).norm();
    CAPTURE(error);
    CHECK(error<tol);
  }
  SECTION("Least Squares and interpolation of functions should be similar"){
    Eigen::VectorXd xdata = Eigen::ArrayXd::LinSpaced(100,0,1);
    Eigen::VectorXd ydata = xdata.array().sin().matrix();
    ChebTools::ChebyshevExpansion chebLS = ChebTools::ChebyshevExpansion::cheb_from_leastSquares(10, xdata, ydata);
    ChebTools::ChebyshevExpansion chebf = ChebTools::ChebyshevExpansion::factory(10,sin1,0, 1);
    error = (chebf.coef()-chebLS.coef()).norm();
    CAPTURE(error);
    CHECK(error<tol);
  }
  SECTION("Cubic Least squares"){
    Eigen::VectorXd xdata = Eigen::ArrayXd::LinSpaced(100,-1,1);
    Eigen::VectorXd ydata = xdata.array()*xdata.array()*xdata.array();
    xdata(1) = 0; xdata(2) = 0;
    ydata(1) = -.01; ydata(2) = .01;
    ChebTools::ChebyshevExpansion chebLS = ChebTools::ChebyshevExpansion::cheb_from_leastSquares(3, xdata, ydata);
    ChebTools::ChebyshevExpansion chebf = ChebTools::ChebyshevExpansion::from_powxn(3,-1,1);
    error = (chebf.coef()-chebLS.coef()).norm();
    CAPTURE(error);
    CHECK(error<tol);
  }
}
