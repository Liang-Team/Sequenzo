#include <pybind11/pybind11.h>
#include "OMdistance.cpp"
#include "OMlocDistance.cpp"
#include "OMspellDistance.cpp"
#include "OMtspellDistance.cpp"
#include "OMspellNewDistance.cpp"
#include "OMslenDistance.cpp"
#include "dist2matrix.cpp"
#include "DHDdistance.cpp"
#include "CHI2distance.cpp"
#include "LCPdistance.cpp"
#include "LCPspellDistance.cpp"
#include "LCPmstDistance.cpp"
#include "LCPprodDistance.cpp"
#include "LCSdistance.cpp"
#include "SVRspellDistance.cpp"
#include "NMSdistance.cpp"
#include "NMSMSTdistance.cpp"
#include "NMSMSTSoftdistanceII.cpp"
#include "TWEDdistance.cpp"

namespace py = pybind11;

PYBIND11_MODULE(c_code, m) {
    py::class_<dist2matrix>(m, "dist2matrix")
            .def(py::init<int, py::array_t<int>, py::array_t<double>>())
            .def("padding_matrix", &dist2matrix::padding_matrix);

    py::class_<LCPdistance>(m, "LCPdistance")
            .def(py::init<py::array_t<int>, int, int, py::array_t<int>>())
            .def("compute_all_distances", &LCPdistance::compute_all_distances)
            .def("compute_refseq_distances", &LCPdistance::compute_refseq_distances);

    py::class_<CHI2distance>(m, "CHI2distance")
            .def(py::init<py::array_t<double>, py::array_t<double>, double, py::array_t<int>>())
            .def("compute_all_distances", &CHI2distance::compute_all_distances)
            .def("compute_refseq_distances", &CHI2distance::compute_refseq_distances);

    py::class_<LCSdistance>(m, "LCSdistance")
            .def(py::init<py::array_t<int>, py::array_t<int>, int, py::array_t<int>>())
            .def("compute_all_distances", &LCSdistance::compute_all_distances)
            .def("compute_refseq_distances", &LCSdistance::compute_refseq_distances);

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

    py::class_<OMtspellDistance>(m, "OMtspellDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, double, py::array_t<double>, py::array_t<double>, py::array_t<int>, py::array_t<double>>())
            .def("compute_all_distances", &OMtspellDistance::compute_all_distances)
            .def("compute_refseq_distances", &OMtspellDistance::compute_refseq_distances);

    py::class_<OMspellNewDistance>(m, "OMspellNewDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, double, py::array_t<double>, py::array_t<double>, py::array_t<int>>())
            .def("compute_all_distances", &OMspellNewDistance::compute_all_distances)
            .def("compute_refseq_distances", &OMspellNewDistance::compute_refseq_distances);

    py::class_<OMslenDistance>(m, "OMslenDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, py::array_t<double>, py::array_t<double>, int, py::array_t<int>>())
            .def("compute_all_distances", &OMslenDistance::compute_all_distances)
            .def("compute_refseq_distances", &OMslenDistance::compute_refseq_distances);

    py::class_<OMdistance>(m, "OMdistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, py::array_t<int>, py::array_t<double>>(),
                 py::arg("sequences"), py::arg("sm"), py::arg("indel"), py::arg("norm"), py::arg("seqlength"), py::arg("refseqS"),
                 py::arg("indellist") = py::array_t<double>())
            .def("compute_all_distances", &OMdistance::compute_all_distances)
            .def("compute_refseq_distances", &OMdistance::compute_refseq_distances);

    py::class_<OMlocDistance>(m, "OMlocDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, py::array_t<int>, double, double>())
            .def("compute_all_distances", &OMlocDistance::compute_all_distances)
            .def("compute_refseq_distances", &OMlocDistance::compute_refseq_distances);

    py::class_<SVRspellDistance>(m, "SVRspellDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, py::array_t<int>, py::array_t<double>, py::array_t<double>, int, py::array_t<int>>())
            .def("compute_all_distances", &SVRspellDistance::compute_all_distances)
            .def("compute_refseq_distances", &SVRspellDistance::compute_refseq_distances);

    py::class_<NMSdistance>(m, "NMSdistance")
            .def(py::init<py::array_t<int>, py::array_t<int>, py::array_t<double>, int, py::array_t<int>>())
            .def("compute_all_distances", &NMSdistance::compute_all_distances)
            .def("compute_refseq_distances", &NMSdistance::compute_refseq_distances);

    py::class_<NMSMSTdistance>(m, "NMSMSTdistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, py::array_t<int>, py::array_t<double>, int, py::array_t<int>>())
            .def("compute_all_distances", &NMSMSTdistance::compute_all_distances)
            .def("compute_refseq_distances", &NMSMSTdistance::compute_refseq_distances);

    py::class_<NMSMSTSoftdistanceII>(m, "NMSMSTSoftdistanceII")
            .def(py::init<py::array_t<int>, py::array_t<int>, py::array_t<double>, py::array_t<double>, int, py::array_t<int>>())
            .def("compute_all_distances", &NMSMSTSoftdistanceII::compute_all_distances)
            .def("compute_refseq_distances", &NMSMSTSoftdistanceII::compute_refseq_distances);

    py::class_<TWEDdistance>(m, "TWEDdistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, double, double, py::array_t<int>, py::array_t<int>>())
            .def("compute_all_distances", &TWEDdistance::compute_all_distances)
            .def("compute_refseq_distances", &TWEDdistance::compute_refseq_distances);
}