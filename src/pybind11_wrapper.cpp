#include "ChebTools/ChebTools.h"
#include "ChebTools/speed_tests.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace ChebTools;

void init_ChebTools(py::module &m){

    m.def("mult_by", &mult_by);
    m.def("mult_by_inplace", &mult_by_inplace);
    m.def("evaluation_speed_test", &evaluation_speed_test);
    m.def("eigs_speed_test", &eigs_speed_test);
    m.def("eigenvalues", &eigenvalues);
    m.def("eigenvalues_upperHessenberg", &eigenvalues_upperHessenberg);
    m.def("generate_Chebyshev_expansion", &ChebyshevExpansion::factory<std::function<double(double)> >);
    m.def("Eigen_nbThreads", []() { return Eigen::nbThreads(); });
    m.def("Eigen_setNbThreads", [](int Nthreads) { return Eigen::setNbThreads(Nthreads); });
    py::class_<ChebyshevExpansion>(m, "ChebyshevExpansion")
        .def(py::init<const std::vector<double> &, double, double>())
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self + double())
        .def(py::self - double())
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self *= double())
        .def(py::self * py::self)
        .def("times_x", &ChebyshevExpansion::times_x)
        .def("times_x_inplace", &ChebyshevExpansion::times_x_inplace)
        //.def("__repr__", &Vector2::toString);
        .def("coef", &ChebyshevExpansion::coef)
        .def("companion_matrix", &ChebyshevExpansion::companion_matrix)
        .def("y", (vectype(ChebyshevExpansion::*)(const vectype &)) &ChebyshevExpansion::y)
        .def("y", (double (ChebyshevExpansion::*)(const double)) &ChebyshevExpansion::y)
        .def("y_Clenshaw", &ChebyshevExpansion::y_Clenshaw)
        .def("real_roots", &ChebyshevExpansion::real_roots)
        .def("real_roots_time", &ChebyshevExpansion::real_roots_time)
        .def("real_roots_approx", &ChebyshevExpansion::real_roots_approx)
        .def("subdivide", &ChebyshevExpansion::subdivide)
        .def("real_roots_intervals", &ChebyshevExpansion::real_roots_intervals)
        .def("deriv", &ChebyshevExpansion::deriv)
        .def("xmin", &ChebyshevExpansion::xmin)
        .def("xmax", &ChebyshevExpansion::xmax)
        .def("get_nodes_n11", &ChebyshevExpansion::get_nodes_n11)
        .def("get_node_function_values", &ChebyshevExpansion::get_node_function_values)
        ;
    py::class_<ChebyshevExpansion2D>(m,"ChebyshevExpansion2D")
        .def(py::init<const std::vector<ChebyshevExpansion> &, const std::vector<ChebyshevExpansion> &, double, double, double, double>())
        .def("max_ydegree", &ChebyshevExpansion2D::max_ydegree)
        .def("max_xdegree", &ChebyshevExpansion2D::max_xdegree)
        .def("xchebs", &ChebyshevExpansion2D::xchebs)
        .def("ychebs", &ChebyshevExpansion2D::ychebs)
        .def("addExpansions", &ChebyshevExpansion2D::addExpansions)
        .def("z_recurrence", &ChebyshevExpansion2D::z_recurrence)
        .def("z_Clenshaw", &ChebyshevExpansion2D::z_Clenshaw)
        .def("z", &ChebyshevExpansion2D::z)
        .def("chebExpansion_atx", &ChebyshevExpansion2D::chebExpansion_atx)
        .def("chebExpansion_aty", &ChebyshevExpansion2D::chebExpansion_aty)
        .def("pivots_from_factory", &ChebyshevExpansion2D::pivots_from_factory)
        .def("generate_Chebyshev_expansion2d", &ChebyshevExpansion2D::factory)
        .def("construct_MatrixPolynomial_inx", &ChebyshevExpansion2D::construct_MatrixPolynomial_inx)
        .def("construct_MatrixPolynomial_iny", &ChebyshevExpansion2D::construct_MatrixPolynomial_iny)
        //.def("common_roots",(std::vector<Eigen::Vector2d>(ChebyshevExpansion2D::*)(const ChebyshevExpansion2D &, const ChebyshevExpansion2D &, bool)), &ChebyshevExpansion2D::common_roots)
        //.def("common_roots",(std::vector<Eigen::Vector2d>(ChebyshevExpansion2D::*)(int, int, std::function<double(double,double)>, std::function<double(double,double)>, double, double, double, double, bool)), &ChebyshevExpansion2D::common_roots)
        ;
}

PYBIND11_PLUGIN(ChebTools) {
    py::module m("ChebTools", "C++ tools for working with Chebyshev expansions");
    init_ChebTools(m);
    return m.ptr();
}
