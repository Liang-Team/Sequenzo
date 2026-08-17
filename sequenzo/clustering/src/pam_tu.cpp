#include "pam_bindings.h"

#include "PAM.cpp"
#include "PAMonce.cpp"

namespace py = pybind11;

void register_pam_engines(py::module_& module) {
    py::class_<PAM>(module, "PAM")
        .def(py::init<int, PamDoubleArray, PamIntArray, int,
                      PamDoubleArray>(),
             py::arg("nelements"),
             py::arg("diss").noconvert(),
             py::arg("centroids").noconvert(),
             py::arg("npass"),
             py::arg("weights").noconvert())
        .def("runclusterloop", &PAM::runclusterloop)
        .def("runclusterloop_one_based", &PAM::runclusterloop_one_based);

    py::class_<PAMonce>(module, "PAMonce")
        .def(py::init<int, PamDoubleArray, PamIntArray, int,
                      PamDoubleArray>(),
             py::arg("nelements"),
             py::arg("diss").noconvert(),
             py::arg("centroids").noconvert(),
             py::arg("npass"),
             py::arg("weights").noconvert())
        .def("runclusterloop", &PAMonce::runclusterloop)
        .def("runclusterloop_one_based", &PAMonce::runclusterloop_one_based)
        .def("diagnostics", &PAMonce::diagnostics);
}
