#include "pam_bindings.h"

#include "PAM.cpp"
#include "PAMonce.cpp"

namespace py = pybind11;

void register_pam_engines(py::module_& module) {
    py::class_<PAM>(module, "PAM")
        .def(py::init<int, PamDoubleArray, PamIntArray, int,
                      PamDoubleArray, int, std::size_t>(),
             py::arg("nelements"),
             py::arg("diss").noconvert(),
             py::arg("centroids").noconvert(),
             py::arg("npass"),
             py::arg("weights").noconvert(),
             py::arg("requested_threads") = 0,
             py::arg("memory_budget_bytes") = 0)
        .def("runclusterloop", &PAM::runclusterloop)
        .def("runclusterloop_one_based", &PAM::runclusterloop_one_based)
        .def("objective", &PAM::objective)
        .def("diagnostics", &PAM::diagnostics);

    py::class_<PAMonce>(module, "PAMonce")
        .def(py::init<int, py::array, PamIntArray, int,
                      PamDoubleArray, int, std::size_t>(),
             py::arg("nelements"),
             py::arg("diss").noconvert(),
             py::arg("centroids").noconvert(),
             py::arg("npass"),
             py::arg("weights").noconvert(),
             py::arg("requested_threads") = 0,
             py::arg("memory_budget_bytes") = 0)
        .def("runclusterloop", &PAMonce::runclusterloop)
        .def("runclusterloop_one_based", &PAMonce::runclusterloop_one_based)
        .def("objective", &PAMonce::objective)
        .def("set_collect_diagnostics", &PAMonce::set_collect_diagnostics)
        .def("build_initial_medoids", &PAMonce::build_initial_medoids)
        .def("diagnostics", &PAMonce::diagnostics)
        .def(py::init<int, py::array, PamIntArray, int,
                      PamDoubleArray, int, std::size_t, PamIntArray>(),
             py::arg("nelements"),
             py::arg("diss").noconvert(),
             py::arg("centroids").noconvert(),
             py::arg("npass"),
             py::arg("weights").noconvert(),
             py::arg("requested_threads"),
             py::arg("memory_budget_bytes"),
             py::arg("build_tie_keys").noconvert())
        .def(py::init<int, py::array, PamIntArray, int,
                      PamDoubleArray, int, std::size_t, PamIntArray,
                      PamDoubleArray>(),
             py::arg("nelements"),
             py::arg("diss").noconvert(),
             py::arg("centroids").noconvert(),
             py::arg("npass"),
             py::arg("weights").noconvert(),
             py::arg("requested_threads"),
             py::arg("memory_budget_bytes"),
             py::arg("build_tie_keys").noconvert(),
             py::arg("distance_codebook").noconvert());
}
