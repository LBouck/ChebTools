#ifndef CHEBTOOLS_H
#define CHEBTOOLS_H

#include "Eigen/Dense"
#include <vector>
#include <iostream>

namespace ChebTools{

    typedef Eigen::VectorXd vectype;

    const Eigen::VectorXd &get_extrema(std::size_t N);

    Eigen::VectorXcd eigenvalues(const Eigen::MatrixXd &A, bool balance);
    Eigen::VectorXd eigenvalues_upperHessenberg(const Eigen::MatrixXd &A, bool balance);

    class ChebyshevExpansion {
    private:
        vectype m_c;
        double m_xmin, m_xmax;

        vectype m_recurrence_buffer;
        Eigen::MatrixXd m_recurrence_buffer_matrix;
        void resize() {
            m_recurrence_buffer.resize(m_c.size());
        }

        //reduce_zeros changes the m_c field so that our companion matrix doesnt have nan values in it
        //all this does is truncate m_c such that there are no trailing zero values
        static Eigen::VectorXd reduce_zeros(const Eigen:: VectorXd &chebCoeffs){
          //these give us a threshold for what coefficients are large enough
          double largeTerm = 1e-15;
          if (chebCoeffs.size()>=1 && std::abs(chebCoeffs(0))>largeTerm){
            largeTerm = chebCoeffs(0);
          }
          //if the second coefficient is larger than the first, then make our tolerance
          //based on the second coefficient, this is useful for functions whose mean value
          //is zero on the interval
          if (chebCoeffs.size()>=2 && std::abs(chebCoeffs(1))>largeTerm){
            largeTerm = chebCoeffs(1);
          }
          double tol = largeTerm*(1e-15);
          double neededSize = chebCoeffs.size();
          //loop over m_c backwards, if we run into large enough coefficient, then record the size and break
          for (int i=chebCoeffs.size()-1;i>=0;i--){
            if (std::abs(chebCoeffs(i))>tol){
              neededSize = i+1;
              break;
            }
            neededSize--;
          }
          //neededSize gives us the number of coefficients that are nonzero
          //we will resize m_c such that there are essentially no trailing zeros
          return chebCoeffs.head(neededSize);
        }

    public:

        ChebyshevExpansion(const vectype &c, double xmin = -1, double xmax = 1) : m_c(c), m_xmin(xmin), m_xmax(xmax) { resize(); };
        ChebyshevExpansion(const std::vector<double> &c, double xmin = -1, double xmax = 1) : m_xmin(xmin), m_xmax(xmax) {
            m_c = Eigen::Map<const Eigen::VectorXd>(&(c[0]), c.size());
            resize();
        };
        double xmin(){ return m_xmin; }
        double xmax(){ return m_xmax; }

        // Move constructor (C++11 only)
        ChebyshevExpansion(const vectype &&c, double xmin = -1, double xmax = 1) : m_c(c), m_xmin(xmin), m_xmax(xmax) { resize(); };

        ChebyshevExpansion operator+(const ChebyshevExpansion &ce2) const ;
        ChebyshevExpansion& operator+=(const ChebyshevExpansion &donor);
        ChebyshevExpansion operator*(double value) const;
        ChebyshevExpansion operator+(double value) const;
        ChebyshevExpansion operator-(double value) const;
        ChebyshevExpansion& operator*=(double value);
        ChebyshevExpansion& operator+=(double value);
        ChebyshevExpansion& operator-=(double value);
        /*
         * @brief Multiply two Chebyshev expansions together; thanks to Julia code from Bradley Alpert, NIST
         *
         * Convertes padded expansions to nodal functional values, functional values are multiplied together,
         * and then inverse transformation is used to return to coefficients of the product
         */
        ChebyshevExpansion operator*(const ChebyshevExpansion &ce2) const;
        /*
         * @brief Multiply a Chebyshev expansion by its independent variable \f$x\f$
         */
        ChebyshevExpansion times_x() const;

        ChebyshevExpansion& times_x_inplace();

        /// Friend function that allows for pre-multiplication by a constant value
        friend ChebyshevExpansion operator*(double value, const ChebyshevExpansion &ce){
            return ChebyshevExpansion(std::move(ce.coef()*value),ce.m_xmin, ce.m_xmax);
        };

        /// Get the vector of coefficients
        const vectype &coef() const ;
        /**
        * @brief Do a single input/single output evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
        * @param x A value scaled in the domain [xmin,xmax]
        */
        double y_recurrence(const double x);
        double y_Clenshaw(const double x) const;
        /**
        * @brief Do a vectorized evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
        * @param x A vectype of values in the domain [xmin,xmax]
        */
        vectype y(const vectype &x) const ;
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
        vectype y_recurrence_xscaled(const vectype &xscaled) const ;
        vectype y_Clenshaw_xscaled(const vectype &xscaled) const ;

        /**
        * @brief Construct and return the companion matrix of the Chebyshev expansion
        * @returns A The companion matrix of the expansion
        *
        * See Boyd, SIAM review, 2013, http://dx.doi.org/10.1137/110838297, Appendix A.2
        */
        Eigen::MatrixXd companion_matrix(const Eigen::VectorXd &coeffs) const ;
        /**
        * @brief Return the real roots of the Chebyshev expansion
        * @param only_in_domain If true, only real roots that are within the domain
        *                       of the expansion will be returned, otherwise all real roots
        *
        * The roots are obtained based on the fact that the eigenvalues of the
        * companion matrix are the roots of the Chebyshev expansion.  Thus
        * this function is relatively slow, because an eigenvalue solve is required,
        * which takes O(n^3) FLOPs.  But it is numerically rather reliable.
        *
        * As the order of the expansion increases, the eigenvalue solver in Eigen becomes
        * progressively less and less able to obtain the roots properly. The eigenvalue
        * solver in numpy tends to be more reliable.
        */
        std::vector<double> real_roots(bool only_in_domain = true) const ;
        std::vector<ChebyshevExpansion> subdivide(std::size_t Nintervals, std::size_t Norder) const ;
        static std::vector<double> real_roots_intervals(const std::vector<ChebyshevExpansion> &segments, bool only_in_domain = true);

        double real_roots_time(long N);
        std::vector<double> real_roots_approx(long Npoints);

        //std::string toString() const {
        //    return "[" + std::to_string(x) + ", " + std::to_string(y) + "]";
        //}

        static ChebyshevExpansion factoryf(const int N, const Eigen::VectorXd &f, const double xmin, const double xmax) ;

        /**
        * @brief Given a callable function, construct the N-th order Chebyshev expansion in [xmin, xmax]
        * @param N The order of the expansion; there will be N+1 coefficients
        * @param func A callable object, taking the x value (in [xmin,xmax]) and returning the y value
        * @param xmin The minimum x value for the fit
        * @param xmax The maximum x value for the fit
        *
        * See Boyd, SIAM review, 2013, http://dx.doi.org/10.1137/110838297, Appendix A.
        */
        template<class double_function>
        static ChebyshevExpansion factory(const int N, double_function func, const double xmin, const double xmax)
        {
            // Get the precalculated extrema values
            const Eigen::VectorXd & x_extrema_n11 = get_extrema(N);

            // Step 1&2: Grid points functional values (function evaluated at the
            // extrema of the Chebyshev polynomial of order N - there are N+1 of them)
            Eigen::VectorXd f(N + 1);
            for (int k = 0; k <= N; ++k) {
                // The extrema in [-1,1] scaled to real-world coordinates
                double x_k = ((xmax - xmin)*x_extrema_n11(k) + (xmax + xmin)) / 2.0;
                f(k) = func(x_k);
            }
            return factoryf(N, f, xmin, xmax);
        };

        /// Convert a monomial term in the form \f$x^n\f$ to a Chebyshev expansion
        static ChebyshevExpansion from_powxn(const std::size_t n, const double xmin, const double xmax);

        template<class vector_type>
        static ChebyshevExpansion from_polynomial(vector_type c, const double xmin, const double xmax) {
            vectype c0(1); c0 << 0;
            ChebyshevExpansion s(c0, xmin, xmax);
            for (std::size_t i = 0; i < c.size(); ++i) {
                s += c(i)*from_powxn(i, xmin, xmax);
            }
            return s;
        }
        /// Return the N-th derivative of this expansion, where N must be >= 1
        ChebyshevExpansion deriv(std::size_t Nderiv) const ;

        /// Get the Chebyshev-Lobatto nodes in the domain [-1,1]
        Eigen::VectorXd get_nodes_n11();
		/// Get the Chebyshev-Lobatto nodes in the domain [xmin, xmax]
		Eigen::VectorXd get_nodes_realworld();
        /// Values of the function at the Chebyshev-Lobatto nodes
        Eigen::VectorXd get_node_function_values();
    };




    /*ChebyshevExpansion2D is a class to create a 2D Chebyshev expansion of a 2d scalar function
    *@field two vectors x_chebs and y_chebs, which contain 1d Chebyshev expansions with respect to x and y
    *Order matters on these vectors as the Chebyshev expansion will be represented by \sum{i,j=0}^{N} T_j(x)T_i(y)
    * where T_j and T_i are ChebyshevExpansion with respect to x and y.
    * @field double x_min,x_max,y_min,y_max are the bounds on the rectangle we are evaluating the ChebyshevExpansion2D on.*/
    class ChebyshevExpansion2D{
    private:
      std::vector<ChebyshevExpansion> x_chebs, y_chebs;
      double x_min,x_max,y_min,y_max;
    public:
      //public constructor
      ChebyshevExpansion2D(const std::vector<ChebyshevExpansion> &xchebs,
                            const std::vector<ChebyshevExpansion> &ychebs,
                            double xmin = -1, double xmax = 1,
                            double ymin = -1, double ymax = 1) :
                            x_chebs(xchebs), y_chebs(ychebs),
                            x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax) {
                              if (x_chebs.size()!=y_chebs.size()){
                                throw std::invalid_argument("Must have same number of y and x ChebyshevExpansions!");
                              }
                              for (int i=0;i<x_chebs.size();i++){
                                if (std::abs(x_chebs.at(i).xmin()-x_min)>1e-15 || std::abs(x_chebs.at(i).xmax()-x_max)>1e-15){
                                  throw std::invalid_argument("x_cheb bounds must match bounds in 2D cheb!");
                                }
                                if (std::abs(y_chebs.at(i).xmin()-y_min)>1e-15 || std::abs(y_chebs.at(i).xmax()-y_max)>1e-15){\
                                  throw std::invalid_argument("y_cheb bounds must match bounds in 2D cheb!");
                                }
                              }
                            };
      //getter member functions to retrieve private fields
      double xmin(){ return x_min; }
      double xmax(){ return x_max; }
      double ymin(){ return y_min; }
      double ymax(){ return y_max; }

      //we return references for x_chebs and y_chebs so no copies are made
      const std::vector<ChebyshevExpansion> &xchebs() const { return x_chebs; }
      const std::vector<ChebyshevExpansion> &ychebs() const { return y_chebs; }

      //add 1d chebs to 2d expansions
      void addExpansions(ChebyshevExpansion xCheb, ChebyshevExpansion yCheb){
        if (std::abs(xCheb.xmin()-x_min)>1e-15 || std::abs(xCheb.xmax()-x_max)>1e-15){
          throw std::invalid_argument("x_cheb bounds must match bounds in 2D cheb!");
        }
        if (std::abs(yCheb.xmin()-y_min)>1e-15 || std::abs(yCheb.xmax()-y_max)>1e-15){
          throw std::invalid_argument("y_cheb bounds must match bounds in 2D cheb!");
        }
        x_chebs.push_back(xCheb); y_chebs.push_back(yCheb);
      }

      //evaluates the ChebyshevExpansion2D using y_recurrence from ChebyshevExpansion
      double z_recurrence(const double x, const double y){
        double z = 0;
        for (int i=0;i<x_chebs.size();i++){
          z+=x_chebs.at(i).y_recurrence(x)*y_chebs.at(i).y_recurrence(y);
        }
        return z;
      }

      //evaluates the ChebyshevExpansion2D using y_Clenshaw from ChebyshevExpansion
      double z_Clenshaw(const double x, const double y) const{
        double z = 0;
        for (int i=0;i<x_chebs.size();i++){
          z+=x_chebs.at(i).y_Clenshaw(x)*y_chebs.at(i).y_Clenshaw(y);
        }
        return z;
      }

      //vectorized way of evaluating a grid of x and y values
      Eigen::ArrayXXd z(const vectype xs, const vectype ys) const{
        Eigen::ArrayXXd z_array(ys.size(),xs.size());
        for (int i=z_array.rows()-1;i>=0;i--){
          for (int j=0;j<z_array.cols();j++){
            z_array(i,j) = z_Clenshaw(xs(j), ys(i));
          }
        }
        return z_array;
      }
      //computes a companion matrix with respect to the y direction and a given x value
      //this allows to find roots of the ChebyshevExpansion2D at a given x value
      //this will be useful when we start doing more complicated rootfinding
      Eigen::MatrixXd companionMatrix_atx(double x) const{
        std::vector<double> starting_coeffs;
        starting_coeffs.push_back(0);
        ChebyshevExpansion cheb_atx = ChebyshevExpansion(starting_coeffs,y_min,y_max);
        for (int i=0;i<x_chebs.size();i++){
          cheb_atx+= x_chebs.at(i).y_Clenshaw(x)*y_chebs.at(i);
        }
        Eigen::MatrixXd companion = cheb_atx.companion_matrix();
        return companion;
      }

      //computes a companion matrix with respect to the x direction and a given y value
      //this allows to find roots of the ChebyshevExpansion2D at a given y value
      //this will be useful when we start doing more complicated rootfinding
      Eigen::MatrixXd companionMatrix_aty(double y) const{
        std::vector<double> starting_coeffs;
        starting_coeffs.push_back(0);
        ChebyshevExpansion cheb_aty = ChebyshevExpansion(starting_coeffs,y_min,y_max);
        for (int i=0;i<y_chebs.size();i++){
          cheb_aty+= y_chebs.at(i).y_Clenshaw(y)*x_chebs.at(i);
        }
        Eigen::MatrixXd companion = cheb_aty.companion_matrix();
        return companion;
      }

      // TODO: factory,static common roots function
      static Eigen::Vector3d findpivot(Eigen::ArrayXXd, Eigen::VectorXd, Eigen::VectorXd);
      static ChebyshevExpansion2D factory(int, int, std::function<double(double,double)>,double, double, double, double);
      static std::vector<Eigen::Vector2d> common_roots(ChebyshevExpansion2D,ChebyshevExpansion2D);
      static Eigen::MatrixXd bezout_atx(ChebyshevExpansion2D,ChebyshevExpansion2D,double);
      static Eigen::MatrixXd bezout_aty(ChebyshevExpansion2D,ChebyshevExpansion2D,double);

    };

}; /* namespace ChebTools */
#endif
