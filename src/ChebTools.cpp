#include "ChebTools/ChebTools.h"
#include "Eigen/Dense"

#include <algorithm>
#include <functional>

#include <chrono>
#include <iostream>
#include <map>
#include <limits>

#ifndef DBL_EPSILON
#define DBL_EPSILON std::numeric_limits<double>::epsilon()
#endif

#ifndef DBL_MAX
#define DBL_MAX std::numeric_limits<double>::max()
#endif

template <class T> T POW2(T x){ return x*x; }
template <class T> bool inbetween(T bound1, T bound2, T val){ return val > std::min(bound1,bound2) && val < std::max(bound1,bound2); }

/// See python code in https://en.wikipedia.org/wiki/Binomial_coefficient#Binomial_coefficient_in_programming_languages
/// This is a direct translation of that code to C++
double binomialCoefficient(const double n, const double k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    double _k = std::min(k, n - k); //# take advantage of symmetry
    double c = 1;
    for (double i = 0; i < _k; ++i) {
        c *= (n - i) / (i + 1);
    }
    return c;
}

namespace ChebTools {

    inline bool ValidNumber(double x){
        // Idea from http://www.johndcook.com/IEEE_exceptions_in_cpp.html
        return (x <= DBL_MAX && x >= -DBL_MAX);
    };

    void balance_matrix(const Eigen::MatrixXd &A, Eigen::MatrixXd &Aprime, Eigen::MatrixXd &D) {
        // https://arxiv.org/pdf/1401.5766.pdf (Algorithm #3)
        const int p = 2;
        double beta = 2; // Radix base (2?)
        int iter = 0;
        Aprime = A;
        D = Eigen::MatrixXd::Identity(A.rows(), A.cols());
        bool converged = false;
        do {
            converged = true;
            for (Eigen::Index i = 0; i < A.rows(); ++i) {
                double c = Aprime.col(i).lpNorm<p>();
                double r = Aprime.row(i).lpNorm<p>();
                double s = pow(c, p) + pow(r, p);
                double f = 1;
                //if (!ValidNumber(c)){
                //    std::cout << A << std::endl;
                //    throw std::range_error("c is not a valid number in balance_matrix"); }
                //if (!ValidNumber(r)) { throw std::range_error("r is not a valid number in balance_matrix"); }
                while (c < r/beta) {
                    c *= beta;
                    r /= beta;
                    f *= beta;
                }
                while (c >= r*beta) {
                    c /= beta;
                    r *= beta;
                    f /= beta;
                }
                if (pow(c, p) + pow(r, p) < 0.95*s) {
                    converged = false;
                    D(i, i) *= f;
                    Aprime.col(i) *= f;
                    Aprime.row(i) /= f;
                }
            }
            iter++;
            if (iter > 50) {
                break;
            }
        } while (!converged);
    }

    class ChebyshevExtremaLibrary {
    private:
        std::map<std::size_t, Eigen::VectorXd> vectors;
        void build(std::size_t N) {
            double NN = static_cast<double>(N); // just a cast
            vectors[N] = (Eigen::VectorXd::LinSpaced(N + 1, 0, NN).array()*EIGEN_PI / N).cos();
        }
    public:
        const Eigen::VectorXd & get(std::size_t N) {
            auto it = vectors.find(N);
            if (it != vectors.end()) {
                return it->second;
            }
            else {
                build(N);
                return vectors.find(N)->second;
            }
        }
    };
    static ChebyshevExtremaLibrary extrema_library;
    const Eigen::VectorXd &get_extrema(std::size_t N){
        return extrema_library.get(N);
    }

    class ChebyshevRootsLibrary {
    private:
        std::map<std::size_t, Eigen::VectorXd> vectors;
        void build(std::size_t N) {
            double NN = static_cast<double>(N); // just a cast
            vectors[N] = ((Eigen::VectorXd::LinSpaced(N, 0, NN - 1).array() + 0.5)*EIGEN_PI / NN).cos();
        }
    public:
        const Eigen::VectorXd & get(std::size_t N) {
            auto it = vectors.find(N);
            if (it != vectors.end()) {
                return it->second;
            }
            else {
                build(N);
                return vectors.find(N)->second;
            }
        }
    };
    static ChebyshevRootsLibrary roots_library;

    class LMatrixLibrary {
    private:
        std::map<std::size_t, Eigen::MatrixXd> matrices;
        void build(std::size_t N) {
            Eigen::MatrixXd L(N + 1, N + 1); ///< Matrix of coefficients
            for (int j = 0; j <= N; ++j) {
                for (int k = j; k <= N; ++k) {
                    double p_j = (j == 0 || j == N) ? 2 : 1;
                    double p_k = (k == 0 || k == N) ? 2 : 1;
                    L(j, k) = 2.0 / (p_j*p_k*N)*cos((j*EIGEN_PI*k) / N);
                    // Exploit symmetry to fill in the symmetric elements in the matrix
                    L(k, j) = L(j, k);
                }
            }
            matrices[N] = L;
        }
    public:
        const Eigen::MatrixXd & get(std::size_t N) {
            auto it = matrices.find(N);
            if (it != matrices.end()) {
                return it->second;
            }
            else {
                build(N);
                return matrices.find(N)->second;
            }
        }
    };
    static LMatrixLibrary l_matrix_library;

    class UMatrixLibrary {
    private:
        std::map<std::size_t, Eigen::MatrixXd> matrices;
        void build(std::size_t N) {
            Eigen::MatrixXd U(N + 1, N + 1); ///< Matrix of coefficients
            for (int j = 0; j <= N; ++j) {
                for (int k = j; k <= N; ++k) {
                    U(j, k) = cos((j*EIGEN_PI*k) / N);
                    // Exploit symmetry to fill in the symmetric elements in the matrix
                    U(k, j) = U(j, k);
                }
            }
            matrices[N] = U;
        }
    public:
        const Eigen::MatrixXd & get(std::size_t N) {
            auto it = matrices.find(N);
            if (it != matrices.end()) {
                return it->second;
            }
            else {
                build(N);
                return matrices.find(N)->second;
            }
        }
    };
    static UMatrixLibrary u_matrix_library;


    class DiffMatrixLibrary{
    private:
      static Eigen::MatrixXd first_order_diff(std::size_t N){
        // std::cout<<"Inside first_order_diff"<<std::endl;
        Eigen::MatrixXd first_diff_matrix(N+1,N+1);
        Eigen::VectorXd cheb_nodes = get_extrema(N);
        // std::cout<<"Size of cheb_nodes"<<std::endl;
        // std::cout<<cheb_nodes.size()<<std::endl;

        double deltai, deltaj;
        for (std::size_t i=0;i<N+1;i++){
          for (std::size_t j=0;j<N+1;j++){
            deltai = (i==0 || i==N) ? 2 : 1;
            deltaj = (j==0 || j==N) ? 2 : 1;
            if (i==j){ first_diff_matrix(i,j) = 0;}
            else{ first_diff_matrix(i,j) = (deltai/deltaj)*std::pow((double) -1,i+j)/(std::cos(i*EIGEN_PI/N)-std::cos(j*EIGEN_PI/N)); }
          }
        }
        // std::cout<<"Diagonal terms"<<std::endl;
        double sum;
        for (std::size_t i=0;i<N+1;i++){
          sum = first_diff_matrix.row(i).sum();
          first_diff_matrix(i,i) = -sum;
        }
        return first_diff_matrix;
      }
      // static Eigen::MatrixXd second_order_diff(std::size_t N){
      //   // std::cout<<"Inside second_order_diff"<<std::endl;
      //   Eigen::MatrixXd first_diff_matrix = first_order_diff(N);
      //   // std::cout<<"Size of first order matrice"<<std::endl;
      //   // std::cout<<first_diff_matrix.rows()<<std::endl;
      //   // std::cout<<first_diff_matrix.cols()<<std::endl;
      //   Eigen::VectorXd cheb_nodes = get_extrema(N);
      //   // std::cout<<"Size of cheb_nodes"<<std::endl;
      //   // std::cout<<cheb_nodes.size()<<std::endl;
      //   Eigen::MatrixXd second_diff_matrix(N+1,N+1);
      //   for (std::size_t i=0;i<N+1;i++){
      //     for (std::size_t j=0;j<N+1;j++){
      //       if (i==j){
      //         if (i!=0){
      //           // std::cout<<"Size of row(i).head(i) "<<first_diff_matrix.row(i).head(i).size()<<std::endl;
      //           // std::cout<<"Size of Eigen::VectorXd::Constant(i,1) "<<Eigen::VectorXd::Constant(i,1).size()<<std::endl;
      //           // std::cout<<"Size of cheb_nodes.head(i) "<<cheb_nodes.head(i).size()<<std::endl;
      //           second_diff_matrix(i,j) = 2*std::pow(first_diff_matrix(i,i),2);
      //           second_diff_matrix(i,j) += 2*(first_diff_matrix.row(i).head(i).array()/
      //                                         (cheb_nodes(i)*Eigen::VectorXd::Constant(i,1).array()-cheb_nodes.head(i).array()).transpose()).sum();
      //         }
      //         if (i!=N){
      //           // std::cout<<"Size of first_diff_matrix.row(i).tail(N-i-1) "<<first_diff_matrix.row(i).tail(N-i-1).size()<<std::endl;
      //           // std::cout<<"Size of Eigen::VectorXd::Constant(N-i-1,1) "<<Eigen::VectorXd::Constant(N-i-1,1).size()<<std::endl;
      //           // std::cout<<"Size of cheb_nodes.tail(N-i-1) "<<cheb_nodes.tail(N-i-1).size()<<std::endl;
      //           Eigen::ArrayXd quotient = cheb_nodes(i)*Eigen::VectorXd::Constant(N-i-1,1).array()-cheb_nodes.tail(N-i-1).array();
      //           // std::cout<<"Done with quotient"<<std::endl;
      //           // std::cout<<"Dimensions of quotient: "<<quotient.rows()<<", "<<quotient.cols()<<std::endl;
      //           // std::cout<<"Dimensions of row: "<<first_diff_matrix.row(i).tail(N-i-1).rows()<<", "<<first_diff_matrix.row(i).tail(N-i-1).cols()<<std::endl;
      //           second_diff_matrix(i,j) += 2*(first_diff_matrix.row(i).tail(N-i-1).array()/quotient.transpose()).sum();
      //         }
      //       }
      //       else{
      //         second_diff_matrix(i,j) = 2*first_diff_matrix(i,j)*(first_diff_matrix(i,i)-1/(cheb_nodes(i)-cheb_nodes(j)));
      //       }
      //     }
      //   }
      //   return second_diff_matrix;
      // }
      public:
        static Eigen::MatrixXd norder_diff_matrix(int order, std::size_t N){
          if (order==1){
            return first_order_diff(N);
          }
          // else if (order==2){
          //   return second_order_diff(N);
          // }
          // Eigen::MatrixXd second_diff_matrix = second_order_diff(N);
          Eigen::MatrixXd first_diff_matrix = first_order_diff(N);
          // std::cout<<"Size of second and first order matrices"<<std::endl;
          // std::cout<<second_diff_matrix.rows()<<std::endl;
          // std::cout<<second_diff_matrix.cols()<<std::endl;
          // std::cout<<first_diff_matrix.rows()<<std::endl;
          // std::cout<<first_diff_matrix.cols()<<std::endl;
          int order_left = order;
          Eigen::MatrixXd diff_matrix = Eigen::MatrixXd::Identity(N+1,N+1);
          while (order_left>0){
            // if (order_left-2>0){
            //   diff_matrix *= second_diff_matrix;
            //   order_left -= 2;
            // }
            // else{
              diff_matrix *= first_diff_matrix;
              order_left--;
            // }
          }
          return diff_matrix;
        }
    };
    static DiffMatrixLibrary DiffMatrixLibrary;

    class LeastSquaresMatrixLibrary{
    public:
      static Eigen::MatrixXd leastSquaresMatrix(std::size_t degree_of_cheb, const Eigen::VectorXd &xvals){
        Eigen::MatrixXd ls_mat(xvals.size(),degree_of_cheb+1);
        double xmin = xvals.minCoeff(); double xmax = xvals.maxCoeff();
        std::vector<double> coeffs = {1};
        ChebyshevExpansion jcheb = ChebyshevExpansion(coeffs,xmin,xmax);
        for (std::size_t j=0;j<degree_of_cheb+1;j++){
          if (j>0){ coeffs[j-1] = 0; coeffs.push_back(1); }
           jcheb = ChebyshevExpansion(coeffs,xmin,xmax);
          for (std::size_t i=0;i<xvals.size();i++){
            ls_mat(i,j) = jcheb.y(xvals(i));
          }
        }
        return ls_mat;
      }
    };
    static LeastSquaresMatrixLibrary LeastSquaresMatrixLibrary;
    // From CoolProp
    template<class T> bool is_in_closed_range(T x1, T x2, T x) { return (x >= std::min(x1, x2) && x <= std::max(x1, x2)); };

    ChebyshevExpansion ChebyshevExpansion::operator+(const ChebyshevExpansion &ce2) const {
        if (m_c.size() == ce2.coef().size()) {
            // Both are the same size, nothing creative to do, just add the coefficients
            return ChebyshevExpansion(std::move(ce2.coef() + m_c), m_xmin, m_xmax);
        }
        else{
            if (m_c.size() > ce2.coef().size()) {
                Eigen::VectorXd c(m_c.size()); c.setZero(); c.head(ce2.coef().size()) = ce2.coef();
                return ChebyshevExpansion(c+m_c, m_xmin, m_xmax);
            }
            else {
                std::size_t n = ce2.coef().size();
                Eigen::VectorXd c(n); c.setZero(); c.head(m_c.size()) = m_c;
                return ChebyshevExpansion(c + ce2.coef(), m_xmin, m_xmax);
            }
        }
    };
    ChebyshevExpansion& ChebyshevExpansion::operator+=(const ChebyshevExpansion &donor) {
        std::size_t Ndonor = donor.coef().size(), N1 = m_c.size();
        std::size_t Nmin = std::min(N1, Ndonor), Nmax = std::max(N1, Ndonor);
        // The first Nmin terms overlap between the two vectors
        m_c.head(Nmin) += donor.coef().head(Nmin);
        // If the donor vector is longer than the current vector, resizing is needed
        if (Ndonor > N1) {
            // Resize but leave values as they were
            m_c.conservativeResize(Ndonor);
            // Copy the last Nmax-Nmin values from the donor
            m_c.tail(Nmax - Nmin) = donor.coef().tail(Nmax - Nmin);
        }
        return *this;
    }
    ChebyshevExpansion ChebyshevExpansion::operator*(double value) const {
        return ChebyshevExpansion(m_c*value, m_xmin, m_xmax);
    }
    ChebyshevExpansion ChebyshevExpansion::operator+(double value) const {
        Eigen::VectorXd c = m_c;
        c(0) += value;
        return ChebyshevExpansion(c, m_xmin, m_xmax);
    }
    ChebyshevExpansion ChebyshevExpansion::operator-(double value) const {
        Eigen::VectorXd c = m_c;
        c(0) -= value;
        return ChebyshevExpansion(c, m_xmin, m_xmax);
    }
    ChebyshevExpansion& ChebyshevExpansion::operator*=(double value) {
        m_c *= value;
        return *this;
    }
    ChebyshevExpansion& ChebyshevExpansion::operator+=(double value) {
        m_c(0) += value;
        return *this;
    }
    ChebyshevExpansion& ChebyshevExpansion::operator-=(double value) {
        m_c(0) -= value;
        return *this;
    }
    ChebyshevExpansion ChebyshevExpansion::operator*(const ChebyshevExpansion &ce2) const {

        std::size_t order1 = this->m_c.size()-1,
                    order2 = ce2.coef().size()-1;
        // The order of the product is the sum of the orders of the two expansions
        std::size_t Norder_product = order1 + order2;

        // Create padded vectors, and copy into them the coefficients from this instance
        // and that of the donor
        Eigen::VectorXd a = Eigen::VectorXd::Zero(Norder_product+1),
                        b = Eigen::VectorXd::Zero(Norder_product+1);
        a.head(order1+1) = this->m_c; b.head(order2+1) = ce2.coef();

        // Get the matrices U and V from the libraries
        const Eigen::MatrixXd &U = u_matrix_library.get(Norder_product);
        const Eigen::MatrixXd &V = l_matrix_library.get(Norder_product);

        // Carry out the calculation of the final coefficients
        // U*a is the functional values at the Chebyshev-Lobatto nodes for the first expansion
        // U*b is the functional values at the Chebyshev-Lobatto nodes for the second expansion
        // The functional values are multiplied together in an element-wise sense - this is why both products are turned into arrays
        // The pre-multiplication by V takes us back to coefficients
        return ChebyshevExpansion(V*((U*a).array() * (U*b).array()).matrix(), m_xmin, m_xmax);
    };
    ChebyshevExpansion ChebyshevExpansion::times_x() const {
        // First we treat the of chi*A multiplication in the domain [-1,1]
        Eigen::Index N = m_c.size()-1; // N is the order of A
        Eigen::VectorXd cc(N+2); // Order of x*A is one higher than that of A
        if (N > 1) {
            cc(0) = m_c(1)/2.0;
        }
        if (N > 2) {
            cc(1) = m_c(0) + m_c(2)/2.0;
        }
        for (Eigen::Index i = 2; i < cc.size(); ++i) {
            cc(i) = (i+1 <= N) ? 0.5*(m_c(i-1) + m_c(i+1)) : 0.5*(m_c(i - 1));
        }
        // Scale the values into the real world, which is given by
        // C_scaled = (b-a)/2*(chi*A) + ((b+a)/2)*A
        // where the coefficients in the second term need to be padded with a zero to have
        // the same order as the product of x*A
        Eigen::VectorXd c_padded(N+2); c_padded << m_c, 0;
        Eigen::VectorXd coefs = (((m_xmax - m_xmin)/2.0)*cc).array() + (m_xmax + m_xmin)/2.0*c_padded.array();
        return ChebyshevExpansion(coefs, m_xmin, m_xmax);
    };
    ChebyshevExpansion& ChebyshevExpansion::times_x_inplace() {
        Eigen::Index N = m_c.size() - 1; // N is the order of A
        double diff = ((m_xmax - m_xmin) / 2.0), plus = (m_xmax + m_xmin) / 2.0;
        double cim1old = 0, ciold = 0;
        m_c.conservativeResize(N+2);
        m_c(N+1) = 0.0; // Fill the last entry with a zero
        if (N > 1) {
            // 0-th element
            cim1old = m_c(0); // store the current 0-th element in temporary variable
            m_c(0) = diff*(0.5*m_c(1)) + plus*m_c(0);
        }
        if (N > 2) {
            // 1-th element
            ciold = m_c(1); // store the current 1-th element in temporary variable
            m_c(1) = diff*(cim1old + 0.5*m_c(2)) + plus*m_c(1);
            cim1old = ciold;
        }
        for (Eigen::Index i = 2; i <= N-1; ++i) {
            ciold = m_c(i); // store the current i-th element in temporary variable
            m_c(i) = diff*(0.5*(cim1old + m_c(i + 1)))+plus*m_c(i);
            cim1old = ciold;
        }
        for (Eigen::Index i = N; i <= N + 1; ++i) {
            ciold = m_c(i); // store the current i-th element in temporary variable
            m_c(i) = diff*(0.5*cim1old) + plus*m_c(i);
            cim1old = ciold;
        }
        return *this;
    };
    ChebyshevExpansion ChebyshevExpansion::apply(std::function<Eigen::ArrayXd(const Eigen::ArrayXd &)> &f){
        // 1. Transform Chebyshev-Lobatto node function values by the function f(y) -> y2
        // 2. Go backwards to coefficients from node values c2 = V*y2
        const auto Ndegree = m_c.size()-1;
        const Eigen::MatrixXd &V = l_matrix_library.get(Ndegree);
        return ChebyshevExpansion(V*f(get_node_function_values()).matrix(), xmin(), xmax());
    }

    const vectype &ChebyshevExpansion::coef() const {
        return m_c;
    };
    /**
    * @brief Do a single input/single output evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
    * @param x A value scaled in the domain [xmin,xmax]
    */
    double ChebyshevExpansion::y_recurrence(const double x) {
        // Use the recurrence relationships to evaluate the Chebyshev expansion
        std::size_t Norder = m_c.size() - 1;
        // Scale x linearly into the domain [-1, 1]
        double xscaled = (2 * x - (m_xmax + m_xmin)) / (m_xmax - m_xmin);
        // Short circuit if not using recursive solution
        if (Norder == 0){ return m_c[0]; }
        if (Norder == 1) { return m_c[0] + m_c[1]*xscaled; }

        vectype &o = m_recurrence_buffer;
        o(0) = 1;
        o(1) = xscaled;
        for (int n = 1; n < Norder; ++n) {
            o(n + 1) = 2 * xscaled*o(n) - o(n - 1);
        }
        return m_c.dot(o);
    }
    double ChebyshevExpansion::y_Clenshaw(const double x) const {
        std::size_t Norder = m_c.size() - 1;
        // Scale x linearly into the domain [-1, 1]
        double xscaled = (2 * x - (m_xmax + m_xmin)) / (m_xmax - m_xmin);
        // Short circuit if not using recursive solution
        if (Norder == 0) { return m_c[0]; }
        if (Norder == 1) { return m_c[0] + m_c[1]*xscaled; }

        double u_k = 0, u_kp1 = m_c[Norder], u_kp2 = 0;
        for (int k = static_cast<int>(Norder) - 1; k >= 1; --k) {
            u_k = 2.0*xscaled*u_kp1 - u_kp2 + m_c(k);
            // Update summation values for all but the last step
            if (k > 1) {
                u_kp2 = u_kp1; u_kp1 = u_k;
            }
        }
        return xscaled*u_k - u_kp1 + m_c(0);
    }
    double ChebyshevExpansion::y_Clenshaw_xscaled(const double xscaled) const {
        std::size_t Norder = m_c.size() - 1;
        // Short circuit if not using recursive solution
        if (Norder == 0) { return m_c[0]; }
        if (Norder == 1) { return m_c[0] + m_c[1] * xscaled; }

        double u_k = 0, u_kp1 = m_c[Norder], u_kp2 = 0;
        for (int k = static_cast<int>(Norder) - 1; k >= 1; --k) {
            u_k = 2.0*xscaled*u_kp1 - u_kp2 + m_c(k);
            // Update summation values for all but the last step
            if (k > 1) {
                u_kp2 = u_kp1; u_kp1 = u_k;
            }
        }
        return xscaled*u_k - u_kp1 + m_c(0);
    }
    /**
    * @brief Do a vectorized evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
    * @param x A vectype of values in the domain [xmin,xmax]
    */
    vectype ChebyshevExpansion::y(const vectype &x) const {
        // Scale x linearly into the domain [-1, 1]
        const vectype xscaled = (2 * x.array() - (m_xmax + m_xmin)) / (m_xmax - m_xmin);
        // Then call the function that takes the scaled x values
        return y_recurrence_xscaled(xscaled);
    }
    /**
    * @brief Do a vectorized evaluation of the Chebyshev expansion with the input scaled in the domain [-1,1]
    * @param xscaled A vectype of values scaled to the domain [-1,1] (the domain of the Chebyshev basis functions)
    * @param y A vectype of values evaluated from the expansion
    *
    * By using vectorizable types like Eigen::MatrixXd, without
    * any additional work, "magical" vectorization is happening
    * under the hood, giving a significant speed improvement. From naive
    * testing, the increase was a factor of about 10x.
    */
    vectype ChebyshevExpansion::y_recurrence_xscaled(const vectype &xscaled) const {
        const std::size_t Norder = m_c.size() - 1;

        Eigen::MatrixXd A(xscaled.size(), Norder + 1);

        if (Norder == 0) { return m_c[0]*Eigen::MatrixXd::Ones(A.rows(), A.cols()); }
        if (Norder == 1) { return m_c[0] + m_c[1]*xscaled.array(); }

        // Use the recurrence relationships to evaluate the Chebyshev expansion
        // In this case we do column-wise evaluations of the recurrence rule
        A.col(0).fill(1);
        A.col(1) = xscaled;
        for (int n = 1; n < Norder; ++n) {
            A.col(n + 1).array() = 2 * xscaled.array()*A.col(n).array() - A.col(n - 1).array();
        }
        // In this form, the matrix-vector product will yield the y values
        return A*m_c;
    }
    vectype ChebyshevExpansion::y_Clenshaw_xscaled(const vectype &xscaled) const {
        const std::size_t Norder = m_c.size() - 1;
        vectype u_k, u_kp1(xscaled.size()), u_kp2(xscaled.size());
        u_kp1.fill(m_c[Norder]); u_kp2.fill(0);
        for (int k = static_cast<int>(Norder) - 1; k >= 1; --k) {
            u_k = 2 * xscaled.array()*u_kp1.array() - u_kp2.array() + m_c(k);
            // Update summation values for all but the last step
            if (k > 1) {
                u_kp2 = u_kp1; u_kp1 = u_k;
            }
        }
        return xscaled.array()*u_k.array() - u_kp1.array() + m_c(0);
    }

    Eigen::MatrixXd ChebyshevExpansion::companion_matrix(const Eigen::VectorXd &coeffs) const {
        Eigen::VectorXd new_mc = reduce_zeros(coeffs);
        std::size_t Norder = new_mc.size() - 1;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Norder, Norder);
        // c_wrap wraps the first 0...Norder elements of the coefficient vector
        Eigen::Map<const Eigen::VectorXd> c_wrap(&(new_mc[0]), Norder);
        // First row
        A(0, 1) = 1;
        // Last row
        A.row(Norder - 1) = -c_wrap / (2.0*new_mc(Norder));
        A(Norder - 1, Norder - 2) += 0.5;
        // All the other rows
        for (int j = 1; j < Norder - 1; ++j) {
            A(j, j - 1) = 0.5;
            A(j, j + 1) = 0.5;
        }
        return A;
    }
    std::vector<double> ChebyshevExpansion::real_roots2(bool only_in_domain) const {
        //vector of roots to be returned
        std::vector<double> roots;

        auto N = m_c.size()-1;
        auto Ndegree_scaled = N*2;
        Eigen::VectorXd xscaled = get_extrema(Ndegree_scaled), yy = y_Clenshaw_xscaled(xscaled);

        // a,b,c can also be obtained by solving the matrix system:
        // [x_k^2, x_k, 1] = [b_k] for k in 1,2,3
        for (auto i = 0; i+2 < Ndegree_scaled+1; i += 2){
            const double &x_1 = xscaled[i + 0], &y_1 = yy[i + 0],
                         &x_2 = xscaled[i + 1], &y_2 = yy[i + 1],
                         &x_3 = xscaled[i + 2], &y_3 = yy[i + 2];
            double d = (x_3 - x_2)*(x_2 - x_1)*(x_3 - x_1);
            double a = ((x_3 - x_2)*y_1 - (x_3 - x_1)*y_2 + (x_2 - x_1)*y_3) / d;
            double b = (-(POW2(x_3) - POW2(x_2))*y_1 + (POW2(x_3) - POW2(x_1))*y_2 - (POW2(x_2) - POW2(x_1))*y_3) / d;
            double c = ((x_3 - x_2)*x_2*x_3*y_1 - (x_3 - x_1)*x_1*x_3*y_2 + (x_2 - x_1)*x_2*x_1*y_3) / d;

            // Discriminant of quadratic
            double D = b*b - 4*a*c;
            if (D >= 0) {
                double root1, root2;
                if (a == 0){
                    // Linear
                    root1 = -c/b;
                    root2 = -1000; // Something outside the domain; we are in scaled coordinates so this will definitely get rejected
                }
                else if (D == 0) { // Unlikely due to numerical precision
                    // Two equal real roots
                    root1 = -b/(2*a);
                    root2 = -1000; // Something outside the domain; we are in scaled coordinates so this will definitely get rejected
                }
                else {
                    // Numerically stable method for solving quadratic
                    // From https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
                    double sqrtD = sqrt(D);
                    if (b >= 0){
                        root1 = (-b - sqrtD)/(2*a);
                        root2 = 2*c/(-b-sqrtD);
                    }
                    else {
                        root1 = 2*c/(-b+sqrtD);
                        root2 = (-b+sqrtD)/(2*a);
                    }
                }
                bool in1 = inbetween(x_1, x_3, root1), in2 = inbetween(x_1, x_3, root2);
                const ChebyshevExpansion &e = *this;
                auto secant = [e](double a, double ya, double b, double yb, double yeps = 1e-14, double xeps = 1e-14) {
                    auto c = b - yb*(b - a) / (yb - ya);
                    auto yc = e.y_Clenshaw_xscaled(c);
                    for (auto i = 0; i < 50; ++i){
                        if (yc*ya > 0) {
                            a=c; ya=yc;
                        }
                        else {
                            b=c; yb=yc;
                        }
                        if (std::abs(b - a) < xeps) { break; }
                        if (std::abs(yc) < yeps){ break; }
                        c = b - yb*(b - a) / (yb - ya);
                        yc = e.y_Clenshaw_xscaled(c);
                    }
                    return c;
                };
                int Nroots_inside = static_cast<int>(in1) + static_cast<int>(in2);
                if (Nroots_inside == 2) {
                    // Split the domain at the midline of the quadratic, polish each root against the underlying expansion
                    double x_m = -b/(2*a), y_m = e.y_Clenshaw_xscaled(x_m);
                    root1 = secant(x_1, y_1, x_m, y_m);
                    root2 = secant(x_m, y_m, x_3, y_3);
                    // Rescale back into real-world values in [xmin,xmax] from [-1,1]
                    roots.push_back(((m_xmax - m_xmin)*root1 + (m_xmax + m_xmin)) / 2.0);
                    roots.push_back(((m_xmax - m_xmin)*root2 + (m_xmax + m_xmin)) / 2.0);
                }
                else if(Nroots_inside == 1) {
                    root1 = secant(x_1, y_1, x_3, y_3);
                    roots.push_back(((m_xmax - m_xmin)*root1 + (m_xmax + m_xmin)) / 2.0);
                }
                else {}
            }
        }
        return roots;
    }
    std::vector<double> ChebyshevExpansion::real_roots(bool only_in_domain) const {
      //vector of roots to be returned
        std::vector<double> roots;
        Eigen::VectorXd new_mc = reduce_zeros(m_c);
        //if the Chebyshev polynomial is just a constant, then there are no roots
        //if a_0=0 then there are infinite roots, but for our purposes, infinite roots doesnt make sense
        if (new_mc.size()<=1){ //we choose <=1 to account for the case of no coefficients
          return roots; //roots is empty
        }

        //if the Chebyshev polynomial is linear, then the only possible root is -a_0/a_1
        //we have this else if block because eigen is not a fan of 1x1 matrices
        else if (new_mc.size()==2){
          double val_n11 = -new_mc(0)/new_mc(1);
          const bool is_in_domain = (val_n11 >= -1.0 && val_n11 <= 1.0);
          // Keep it if it is in domain, or if you just want all real roots
          if (!only_in_domain || is_in_domain) {
              // Rescale back into real-world values in [xmin,xmax] from [-1,1]
              double x = ((m_xmax - m_xmin)*val_n11 + (m_xmax + m_xmin)) / 2.0;
              roots.push_back(x);
          }
        }

        //this for all cases of higher order polynomials
        else{
          // The companion matrix is definitely lower Hessenberg, so we can skip the Hessenberg
          // decomposition, and get the real eigenvalues directly.  These eigenvalues are defined
          // in the domain [-1, 1], but it might also include values outside [-1, 1]
          Eigen::VectorXcd eigs = eigenvalues(companion_matrix(new_mc), /* balance = */ true);


          for (Eigen::Index i = 0; i < eigs.size(); ++i) {
              if (std::abs(eigs(i).imag() / eigs(i).real()) < 1e-15) {
                  double val_n11 = eigs(i).real();
                  const bool is_in_domain = (val_n11 >= -1.0 && val_n11 <= 1.0);
                  // Keep it if it is in domain, or if you just want all real roots
                  if (!only_in_domain || is_in_domain) {
                      // Rescale back into real-world values in [xmin,xmax] from [-1,1]
                      double x = ((m_xmax - m_xmin)*val_n11 + (m_xmax + m_xmin)) / 2.0;
                      roots.push_back(x);
                  }
              }
          }
        }
        return roots;

        //// The companion matrix is definitely lower Hessenberg, so we can skip the Hessenberg
        //// decomposition, and get the real eigenvalues directly.  These eigenvalues are defined
        //// in the domain [-1, 1], but it might also include values outside [-1, 1]
        //Eigen::VectorXd real_eigs = eigenvalues_upperHessenberg(companion_matrix().transpose(), /* balance = */ true);
        //
        //std::vector<double> roots;
        //for (Eigen::Index i = 0; i < real_eigs.size(); ++i){
        //    double val_n11 = real_eigs(i);
        //    const bool is_in_domain = (val_n11 >= -1.0 && val_n11 <= 1.0);
        //    // Keep it if it is in domain, or if you just want all real roots
        //    if (!only_in_domain || is_in_domain){
        //        // Rescale back into real-world values in [xmin,xmax] from [-1,1]
        //        double x = ((m_xmax - m_xmin)*val_n11 + (m_xmax + m_xmin)) / 2.0;
        //        roots.push_back(x);
        //    }
        //}
        //return roots;
    }
    std::vector<ChebyshevExpansion> ChebyshevExpansion::subdivide(std::size_t Nintervals, const std::size_t Norder) const {

        if (Nintervals == 1) {
            return std::vector<ChebyshevExpansion>(1, *this);
        }

        std::vector<ChebyshevExpansion> segments;
        double deltax = (m_xmax - m_xmin) / (Nintervals - 1);

        // Vector of values in the range [-1,1] as roots of a high-order Chebyshev
        double NN = static_cast<double>(Norder);
        Eigen::VectorXd xpts_n11 = (Eigen::VectorXd::LinSpaced(Norder + 1, 0, NN)*EIGEN_PI/NN).array().cos();

        for (std::size_t i = 0; i < Nintervals - 1; ++i) {
            double xmin = m_xmin + i*deltax, xmax = m_xmin + (i + 1)*deltax;
            Eigen::VectorXd xrealworld = ((xmax - xmin)*xpts_n11.array() + (xmax + xmin)) / 2.0;
            segments.push_back(factoryf(Norder, y(xrealworld), xmin, xmax));
        }
        return segments;
    }
    std::vector<double> ChebyshevExpansion::real_roots_intervals(const std::vector<ChebyshevExpansion> &segments, bool only_in_domain) {
        std::vector<double> roots;
        for (auto &seg : segments) {
            const auto segroots = seg.real_roots(only_in_domain);
            roots.insert(roots.end(), segroots.cbegin(), segroots.cend());
        }
        return roots;
    }
    std::vector<double> ChebyshevExpansion::real_roots_approx(long Npoints)
    {
        std::vector<double> roots;
        // Vector of values in the range [-1,1] as roots of a high-order Chebyshev
        Eigen::VectorXd xpts_n11 = (Eigen::VectorXd::LinSpaced(Npoints + 1, 0, Npoints)*EIGEN_PI / Npoints).array().cos();
        // Scale values into real-world values
        Eigen::VectorXd ypts = y_recurrence_xscaled(xpts_n11);
        // Eigen::MatrixXd buf(Npoints+1, 2); buf.col(0) = xpts; buf.col(1) = ypts; std::cout << buf << std::endl;
        for (size_t i = 0; i < Npoints - 1; ++i) {
            // The change of sign guarantees at least one root between indices i and i+1
            double y1 = ypts(i), y2 = ypts(i + 1);
            bool signchange = (std::signbit(y1) != std::signbit(y2));
            if (signchange) {
                double xscaled = xpts_n11(i);

                // Fit a quadratic given three points; i and i+1 bracket the root, so need one more constraint
                // i0 is the leftmost of the three indices that will be used; when i == 0, use
                // indices i,i+1,i+2, otherwise i-1,i,i+1
                size_t i0 = (i >= 1) ? i - 1 : i;
                Eigen::Vector3d r;
                r << ypts(i0), ypts(i0 + 1), ypts(i0 + 2);
                Eigen::Matrix3d A;
                for (std::size_t irow = 0; irow < 3; ++irow) {
                    double _x = xpts_n11(i0 + irow);
                    A.row(irow) << _x*_x, _x, 1;
                }
                // abc holds the coefficients a,b,c for y = a*x^2 + b*x + c
                Eigen::VectorXd abc = A.colPivHouseholderQr().solve(r);
                double a = abc[0], b = abc[1], c = abc[2];

                // Solve the quadratic and find the root you want
                double x1 = (-b + sqrt(b*b - 4 * a*c)) / (2 * a);
                double x2 = (-b - sqrt(b*b - 4 * a*c)) / (2 * a);
                bool x1_in_range = is_in_closed_range(xpts_n11(i), xpts_n11(i + 1), x1);
                bool x2_in_range = is_in_closed_range(xpts_n11(i), xpts_n11(i + 1), x2);

                // Double check that only one root is within the range
                if (x1_in_range && !x2_in_range) {
                    xscaled = x1;
                }
                else if (x2_in_range && !x1_in_range) {
                    xscaled = x2;
                }
                else {
                    xscaled = 1e99;
                }

                // Rescale back into real-world values
                double x = ((m_xmax - m_xmin)*xscaled + (m_xmax + m_xmin)) / 2.0;
                roots.push_back(x);
            }
            else {
                // TODO: locate other roots based on derivative considerations
            }
        }
        return roots;
    }

    /// Chebyshev-Lobatto nodes cos(pi*j/N), j = 0,..., N in the range [-1,1]
    Eigen::VectorXd ChebyshevExpansion::get_nodes_n11() {
        std::size_t N = m_c.size()-1;
        return extrema_library.get(N);
    }
    /// Chebyshev-Lobatto nodes cos(pi*j/N), j = 0,..., N mapped to the range [xmin, xmax]
    Eigen::VectorXd ChebyshevExpansion::get_nodes_realworld() {
        return ((m_xmax - m_xmin)*get_nodes_n11().array() + (m_xmax + m_xmin))*0.5;
    }
    /// Values of the function at the Chebyshev-Lobatto nodes
    Eigen::VectorXd ChebyshevExpansion::get_node_function_values() {
        std::size_t N = m_c.size()-1;
        return u_matrix_library.get(N)*m_c;
    }

    ChebyshevExpansion ChebyshevExpansion::factoryf(const std::size_t N, const Eigen::VectorXd &f, const double xmin, const double xmax) {
        // Step 3: Get coefficients for the L matrix from the library of coefficients
        const Eigen::MatrixXd &L = l_matrix_library.get(N);
        // Step 4: Obtain coefficients from vector - matrix product
        return ChebyshevExpansion(L*f, xmin, xmax);
    }
    ChebyshevExpansion ChebyshevExpansion::from_powxn(const std::size_t n, const double xmin, const double xmax) {
        Eigen::VectorXd c = Eigen::VectorXd::Zero(n + 1);
        for (std::size_t k = 0; k <= n / 2; ++k) {
            std::size_t index = n - 2 * k;
            double coeff = binomialCoefficient(static_cast<double>(n), static_cast<double>(k));
            if (index == 0) {
                coeff /= 2.0;
            }
            c(index) = coeff;
        }
        return pow(2, 1-static_cast<int>(n))*ChebyshevExpansion(c, xmin, xmax);
    }
    ChebyshevExpansion ChebyshevExpansion::deriv(std::size_t Nderiv) const {
        // See Mason and Handscomb, p. 34, Eq. 2.52
        // and example in https ://github.com/numpy/numpy/blob/master/numpy/polynomial/chebyshev.py#L868-L964
        vectype c = m_c;
        for (std::size_t deriv_counter = 0; deriv_counter < Nderiv; ++deriv_counter) {
            std::size_t N = c.size() - 1, ///< Order of the expansion
                        Nd = N - 1; ///< Order of the derivative expansion
            vectype cd(N);
            for (std::size_t r = 0; r <= Nd; ++r) {
                cd(r) = 0;
                for (std::size_t k = r + 1; k <= N; ++k) {
                    // Terms where k-r is odd have values, otherwise, they are zero
                    if ((k - r) % 2 == 1) {
                        cd(r) += 2*k*c(k);
                    }
                }
                // The first term with r = 0 is divided by 2 (the single prime in Mason and Handscomb, p. 34, Eq. 2.52)
                if (r == 0) {
                    cd(r) /= 2;
                }
                // Rescale the values if the range is not [-1,1].  Arrives from the derivative of d(xreal)/d(x_{-1,1})
                cd(r) /= (m_xmax-m_xmin)/2.0;
            }
            if (Nderiv == 1) {
                return ChebyshevExpansion(std::move(cd), m_xmin, m_xmax);
            }
            else{
                c = cd;
            }
        }
        return ChebyshevExpansion(std::move(c), m_xmin, m_xmax);
    };

    Eigen::VectorXd eigenvalues_upperHessenberg(const Eigen::MatrixXd &A, bool balance){
        Eigen::VectorXd roots(A.cols());
        Eigen::RealSchur<Eigen::MatrixXd> schur;

        if (balance) {
            Eigen::MatrixXd Abalanced, D;
            balance_matrix(A, Abalanced, D);
            schur.computeFromHessenberg(Abalanced, Eigen::MatrixXd::Zero(Abalanced.rows(), Abalanced.cols()), false);
        }
        else {
            schur.computeFromHessenberg(A, Eigen::MatrixXd::Zero(A.rows(), A.cols()), false);
        }

        const Eigen::MatrixXd &T = schur.matrixT();
        Eigen::Index j = 0;
        for (int i = 0; i < T.cols(); ++i) {
            if (i+1 < T.cols()-1 && std::abs(T(i+1,i)) > DBL_EPSILON){
                // Nope, this is a 2x2 block, keep moving
                i += 1;
            }
            else{
                // This is a 1x1 block, keep this (real) eigenvalue
                roots(j) = T(i, i);
                j++;
            }
        }
        roots.conservativeResize(j-1);
        return roots;
    }

    Eigen::VectorXcd eigenvalues(const Eigen::MatrixXd &A, bool balance) {
        if (balance) {
            Eigen::MatrixXd Abalanced, D;
            balance_matrix(A, Abalanced, D);
            return Abalanced.eigenvalues();
        }
        else {
            return A.eigenvalues();
        }
    }

    ChebyshevExpansion ChebyshevExpansion::cheb_from_bvp(const std::size_t N, const std::vector<std::function<double(double)>> &lhs_coeffs,
                                        const std::function<double(double)> &rhs_func, const std::vector<double> &left_bc,
                                        const std::vector<double> &right_bc, const double xmin, const double xmax){
      if (left_bc.size()!=3 || right_bc.size()!=3){
        throw new std::invalid_argument("Must specify 3 values for boundary conditions!");
      }
      if (N==0){
        throw new std::invalid_argument("Must specify a nonzero degree for the Chebyshev Expansion!");
      }
      // Assembling the rhs vector
      // std::cout<<"RHS"<<std::endl;
      Eigen::VectorXd cheb_nodes = get_extrema(N);
      Eigen::VectorXd f_vec(N+1);
      std::cout<<f_vec.size()<<std::endl;
      for (std::size_t i=1;i<N;i++){
        f_vec(i) = rhs_func((xmax-xmin)*cheb_nodes(i)/2+(xmax+xmin)/2);
      }
      // Incorporating boundary conditions on for left hand side vector
      f_vec(N) = left_bc[2]; f_vec(0) = right_bc[2];

      // Assembling the left hand side matrix that will be all the differentiation together
      Eigen::MatrixXd left_side_matrix = Eigen::MatrixXd::Zero(N+1,N+1);
      Eigen::MatrixXd temp_matrix(N+1,N+1);
      for (std::size_t i=0;i<lhs_coeffs.size();i++){
        if (i==0){
          temp_matrix = Eigen::MatrixXd::Identity(N+1,N+1);
          for (std::size_t j=0;j<lhs_coeffs.size();j++){
            temp_matrix(j,j) = lhs_coeffs[i]((xmax-xmin)*cheb_nodes(j)/2+(xmax+xmin)/2);
          }
        }
        else{
          temp_matrix = std::pow(2/(xmax-xmin),i)*DiffMatrixLibrary::norder_diff_matrix(i,N);
          for (std::size_t j=0;j<lhs_coeffs.size();j++){
            temp_matrix.row(j) *= lhs_coeffs[i]((xmax-xmin)*cheb_nodes(j)/2+(xmax+xmin)/2);
          }
        }
        // std::cout<<"Coeff Matrix for i="<<i<<std::endl;
        // std::cout<<temp_matrix<<std::endl;
        left_side_matrix += temp_matrix;
      }
      // Assemble the boundary conditions
      // the left boundary condition should effect the top row
      // the right boundary condition should effect the bottom row
      left_side_matrix.row(0) = right_bc[1]*Eigen::MatrixXd::Identity(1,N+1);
      left_side_matrix.row(0) += right_bc[0]*2/(xmax-xmin)*DiffMatrixLibrary::norder_diff_matrix(1,N).row(0);
      left_side_matrix.row(N) = left_bc[1]*Eigen::MatrixXd::Identity(N+1,N+1).row(N);
      left_side_matrix.row(N) += left_bc[0]*2/(xmax-xmin)*DiffMatrixLibrary::norder_diff_matrix(1,N).row(N);

      // std::cout<<"Left side matrix: "<<std::endl;
      // std::cout<<left_side_matrix<<std::endl;
      // std::cout<<"Right side vector"<<std::endl;
      // std::cout<<f_vec<<std::endl;
      //
      // std::cout<<"Solution: "<<left_side_matrix.colPivHouseholderQr().solve(f_vec)<<std::endl;
      return ChebyshevExpansion::factoryf(N, left_side_matrix.colPivHouseholderQr().solve(f_vec), xmin, xmax);
    }


    ChebyshevExpansion ChebyshevExpansion::cheb_from_leastSquares(const std::size_t degree_of_cheb, const Eigen::VectorXd &x_data, const Eigen::VectorXd &y_data){

      // We want the user to give us consistent and non empty data
      if (x_data.size()!=y_data.size() || x_data.size()==0){
        throw new std::invalid_argument("Number of x_data and y_data points must agree and be greater than zero!");
      }
      // we also want them to want a ChebyshevExpansion of at least degree 1
      if (degree_of_cheb==0){
        throw new std::invalid_argument("Must specify a nonzero degree for the Chebyshev Expansion!");
      }
      double xmin = x_data.minCoeff(); double xmax = x_data.maxCoeff();
      Eigen::MatrixXd A = LeastSquaresMatrixLibrary::leastSquaresMatrix(degree_of_cheb,x_data);
      Eigen::MatrixXd At = A.transpose();

      // solving the normal equations
      Eigen::VectorXd coeffs = (At*A).colPivHouseholderQr().solve(At*y_data);
      return ChebyshevExpansion::factoryf(degree_of_cheb, coeffs, xmin, xmax);
    }

}; /* namespace ChebTools */
