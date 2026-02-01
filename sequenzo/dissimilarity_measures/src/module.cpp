#include <pybind11/pybind11.h>
#include "OMdistance.cpp"
#include "OMspellDistance.cpp"
#include "OMspellNewDistance.cpp"
#include "dist2matrix.cpp"
#include "DHDdistance.cpp"
#include "LCPdistance.cpp"
#include "LCPspellDistance.cpp"
#include "LCPmstDistance.cpp"
#include "LCPprodDistance.cpp"

namespace py = pybind11;

PYBIND11_MODULE(c_code, m) {
    py::class_<dist2matrix>(m, "dist2matrix")
            .def(py::init<int, py::array_t<int>, py::array_t<double>>())
            .def("padding_matrix", &dist2matrix::padding_matrix);

    py::class_<LCPdistance>(m, "LCPdistance")
            .def(py::init<py::array_t<int>, int, int, py::array_t<int>>())
            .def("compute_all_distances", &LCPdistance::compute_all_distances)
            .def("compute_refseq_distances", &LCPdistance::compute_refseq_distances);

    py::class_<LCPspellDistance>(m, "LCPspellDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, py::array_t<int>, int, int, py::array_t<int>, double>())
            .def("compute_all_distances", &LCPspellDistance::compute_all_distances)
            .def("compute_refseq_distances", &LCPspellDistance::compute_refseq_distances);

    py::class_<LCPmstDistance>(m, "LCPmstDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, py::array_t<int>, py::array_t<double>, int, int, py::array_t<int>>())
            .def("compute_all_distances", &LCPmstDistance::compute_all_distances)
            .def("compute_refseq_distances", &LCPmstDistance::compute_refseq_distances);

    py::class_<LCPprodDistance>(m, "LCPprodDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, py::array_t<int>, py::array_t<double>, int, int, py::array_t<int>>())
            .def("compute_all_distances", &LCPprodDistance::compute_all_distances)
            .def("compute_refseq_distances", &LCPprodDistance::compute_refseq_distances);

    py::class_<DHDdistance>(m, "DHDdistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, int, double, py::array_t<int>>())
            .def("compute_all_distances", &DHDdistance::compute_all_distances)
            .def("compute_refseq_distances", &DHDdistance::compute_refseq_distances);

    py::class_<OMspellDistance>(m, "OMspellDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, double, py::array_t<double>, py::array_t<double>, py::array_t<int>>())
            .def("compute_all_distances", &OMspellDistance::compute_all_distances)
            .def("compute_refseq_distances", &OMspellDistance::compute_refseq_distances);

    py::class_<OMspellNewDistance>(m, "OMspellNewDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, double, py::array_t<double>, py::array_t<double>, py::array_t<int>>())
            .def("compute_all_distances", &OMspellNewDistance::compute_all_distances)
            .def("compute_refseq_distances", &OMspellNewDistance::compute_refseq_distances);

    py::class_<OMdistance>(m, "OMdistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, py::array_t<int>>())
            .def("compute_all_distances", &OMdistance::compute_all_distances)
            .def("compute_refseq_distances", &OMdistance::compute_refseq_distances);
}