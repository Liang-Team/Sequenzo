#include "PAM.cpp"
#include "KMedoid.cpp"
#include "PAMonce.cpp"
#include "weightedinertia.cpp"

namespace py = pybind11;

PYBIND11_MODULE(clustering_c_code, m) {
    py::class_<PAM>(m, "PAM")
            .def(py::init<int, py::array_t<double>, py::array_t<int>, int, py::array_t<double>>())
            .def("runclusterloop", &PAM::runclusterloop);

    py::class_<KMedoid>(m, "KMedoid")
            .def(py::init<int, py::array_t<double>, py::array_t<int>, int, py::array_t<double>>())
            .def("runclusterloop", &KMedoid::runclusterloop);

    py::class_<PAMonce>(m, "PAMonce")
            .def(py::init<int, py::array_t<double>, py::array_t<int>, int, py::array_t<double>>())
            .def("runclusterloop", &PAMonce::runclusterloop);

    py::class_<weightedinertia>(m, "weightedinertia")
            .def(py::init<py::array_t<double>, py::array_t<int>, py::array_t<double>>())
            .def("tmrWeightedInertiaContrib", &weightedinertia::tmrWeightedInertiaContrib);
}