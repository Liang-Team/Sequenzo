#include <pybind11/pybind11.h>
#include "OMdistance.cpp"
#include "OMspellDistance.cpp"
#include "dist2matrix.cpp"
#include "weightedinertia.cpp"
#include "PAMonce.cpp"
#include "DHDdistance.cpp"

namespace py = pybind11;

PYBIND11_MODULE(c_code, m) {
    py::class_<PAMonce>(m, "PAMonce")
            .def(py::init<int, py::array_t<double>, py::array_t<int>, int, py::array_t<double>>())
            .def("runclusterloop", &PAMonce::runclusterloop);

    py::class_<weightedinertia>(m, "weightedinertia")
            .def(py::init<py::array_t<double>, py::array_t<int>, py::array_t<double>>())
            .def("tmrWeightedInertiaContrib", &weightedinertia::tmrWeightedInertiaContrib);

    py::class_<dist2matrix>(m, "dist2matrix")
            .def(py::init<int, py::array_t<int>, py::array_t<double>>())
            .def("padding_matrix", &dist2matrix::padding_matrix);

    py::class_<DHDdistance>(m, "DHDdistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, int, double, py::array_t<int>>())
            .def("compute_all_distances", &DHDdistance::compute_all_distances)
            .def("compute_refseq_distances", &DHDdistance::compute_refseq_distances);

    py::class_<OMspellDistance>(m, "OMspellDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, double, py::array_t<double>, py::array_t<double>, py::array_t<int>>())
            .def("compute_all_distances", &OMspellDistance::compute_all_distances)
            .def("compute_refseq_distances", &OMspellDistance::compute_refseq_distances);

    py::class_<OMdistance>(m, "OMdistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, py::array_t<int>>())
            .def("compute_all_distances", &OMdistance::compute_all_distances)
            .def("compute_refseq_distances", &OMdistance::compute_refseq_distances);
}