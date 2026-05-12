#include "weightedinertia.cpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(core_distance_c_code, m) {
    py::class_<weightedinertia>(m, "weightedinertia")
        .def(py::init<py::array_t<double>, py::array_t<int>, py::array_t<double>>())
        .def("tmrWeightedInertiaContrib", &weightedinertia::tmrWeightedInertiaContrib);
}

