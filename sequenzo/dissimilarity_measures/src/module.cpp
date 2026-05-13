#include <pybind11/pybind11.h>
#include <string>
#include "OMdistance.cpp"
#include "OMlocDistance.cpp"
#include "OMspellDistance.cpp"
#include "OMtspellDistance.cpp"
#include "OMspellUnitFreeDistance.cpp"
#include "OMslenDistance.cpp"
#include "dist2matrix.cpp"
#include "DHDdistance.cpp"
#include "EUCLIDCategoricalDistance.cpp"
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
#include "preprocess.cpp"
#include "normalization_ElzingaStuder.cpp"

#ifdef _OPENMP
#include <omp.h>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#endif

namespace py = pybind11;

static std::string openmp_version_string() {
#ifdef _OPENMP
    switch (_OPENMP) {
        case 199810: return "OpenMP 1.0";
        case 200203: return "OpenMP 2.0";
        case 200505: return "OpenMP 2.5";
        case 200805: return "OpenMP 3.0";
        case 201107: return "OpenMP 3.1";
        case 201307: return "OpenMP 4.0";
        case 201511: return "OpenMP 4.5";
        case 201811: return "OpenMP 5.0";
        case 202011: return "OpenMP 5.1";
        case 202111: return "OpenMP 5.2";
        default: return "OpenMP unknown";
    }
#else
    return "OpenMP not compiled";
#endif
}

static std::string compiler_string() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return std::string("MSVC ") + std::to_string(_MSC_VER);
#else
    return "unknown compiler";
#endif
}

static py::object openmp_runtime_path() {
#ifdef _OPENMP
#if defined(__unix__) || defined(__APPLE__)
    Dl_info dl_info;
    if (dladdr(reinterpret_cast<const void*>(omp_get_max_threads), &dl_info) != 0 &&
        dl_info.dli_fname != nullptr) {
        return py::str(dl_info.dli_fname);
    }
#endif
#endif
    return py::none();
}

static py::object openmp_runtime_library() {
    py::object path_obj = openmp_runtime_path();
    if (path_obj.is_none()) {
        return py::none();
    }

    std::string path = path_obj.cast<std::string>();
    const std::size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return py::str(path);
    }
    return py::str(path.substr(pos + 1));
}

static py::dict collect_openmp_runtime_info(int requested_threads = 0) {
    py::dict info;
#ifdef _OPENMP
    const int previous_max_threads = omp_get_max_threads();
    if (requested_threads > 0) {
        omp_set_num_threads(requested_threads);
    }

    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    info["_OPENMP"] = true;
    info["openmp_version"] = _OPENMP;
    info["openmp_version_string"] = openmp_version_string();
    info["max_threads"] = omp_get_max_threads();
    info["actual_threads"] = actual_threads;
    info["num_threads_outside_parallel"] = omp_get_num_threads();
    info["thread_limit"] = omp_get_thread_limit();
    info["num_procs"] = omp_get_num_procs();
    info["dynamic"] = static_cast<bool>(omp_get_dynamic());
    if (requested_threads > 0) {
        info["requested_threads"] = requested_threads;
    } else {
        info["requested_threads"] = py::none();
    }
    info["compiler"] = compiler_string();
    info["openmp_runtime_path"] = openmp_runtime_path();
    info["openmp_runtime_library"] = openmp_runtime_library();

    if (requested_threads > 0) {
        omp_set_num_threads(previous_max_threads);
    }
#else
    info["_OPENMP"] = false;
    info["openmp_version"] = py::none();
    info["openmp_version_string"] = openmp_version_string();
    info["max_threads"] = 1;
    info["actual_threads"] = 1;
    info["num_threads_outside_parallel"] = 1;
    info["thread_limit"] = 1;
    info["num_procs"] = 1;
    info["dynamic"] = false;
    if (requested_threads > 0) {
        info["requested_threads"] = requested_threads;
    } else {
        info["requested_threads"] = py::none();
    }
    info["compiler"] = compiler_string();
    info["openmp_runtime_path"] = py::none();
    info["openmp_runtime_library"] = py::none();
#endif
    return info;
}

PYBIND11_MODULE(c_code, m) {
    m.def("_openmp_runtime_info",
          &collect_openmp_runtime_info,
          "Internal helper reporting OpenMP metadata from the loaded C++ extension.",
          py::arg("requested_threads") = 0);

    py::class_<dist2matrix>(m, "dist2matrix")
            .def(py::init<int, py::array_t<int>, py::array_t<double>>())
            .def("padding_matrix", &dist2matrix::padding_matrix)
            .def("padding_condensed", &dist2matrix::padding_condensed);

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

    py::class_<EUCLIDCategoricalDistance>(m, "EUCLIDCategoricalDistance")
            .def(py::init<py::array_t<int>, bool, py::array_t<int>>())
            .def("compute_all_distances", &EUCLIDCategoricalDistance::compute_all_distances)
            .def("compute_original_condensed_distances", &EUCLIDCategoricalDistance::compute_original_condensed_distances)
            .def("compute_refseq_distances", &EUCLIDCategoricalDistance::compute_refseq_distances);

    py::class_<OMspellDistance>(m, "OMspellDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, double, py::array_t<double>, py::array_t<double>, py::array_t<int>, py::array_t<int>>())
            .def("compute_all_distances", &OMspellDistance::compute_all_distances)
            .def("compute_refseq_distances", &OMspellDistance::compute_refseq_distances);

    py::class_<OMtspellDistance>(m, "OMtspellDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, double, py::array_t<double>, py::array_t<double>, py::array_t<int>, py::array_t<double>, py::array_t<int>>())
            .def("compute_all_distances", &OMtspellDistance::compute_all_distances)
            .def("compute_refseq_distances", &OMtspellDistance::compute_refseq_distances);

    py::class_<OMspellUnitFreeDistance>(m, "OMspellUnitFreeDistance")
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, double, py::array_t<double>, py::array_t<double>, py::array_t<int>>())
            .def("compute_all_distances", &OMspellUnitFreeDistance::compute_all_distances)
            .def("compute_refseq_distances", &OMspellUnitFreeDistance::compute_refseq_distances);

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
            .def(py::init<py::array_t<int>, py::array_t<double>, double, int, py::array_t<int>, py::array_t<int>, double, double, py::array_t<double>>(),
                 py::arg("sequences"), py::arg("sm"), py::arg("indel"), py::arg("norm"), py::arg("seqlength"), py::arg("refseqS"),
                 py::arg("expcost"), py::arg("context"),
                 py::arg("indellist") = py::array_t<double>())
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
    
    // Reference-based normalization functions (Elzinga & Studer 2019)
    // We apply a theoretical normalization following Elzinga & Studer (2019),
    // dividing distances by their theoretical maxima to ensure comparability across measures.
    m.def("normalize_distance_matrix_ElzingaStuder", 
          &normalize_distance_matrix_ElzingaStuder,
          "Normalize distance matrix using reference-based normalization (equation 9)",
          py::arg("distance_matrix"),
          py::arg("reference_index"));
    
    m.def("normalize_similarity_from_distance_ElzingaStuder",
          &normalize_similarity_from_distance_ElzingaStuder,
          "Convert normalized distance matrix to similarity matrix (equation 11)",
          py::arg("normalized_distance_matrix"));

    m.def("find_unique_sequences",
          &find_unique_sequences,
          "Find unique sequences and build index mapping",
          py::arg("sequences"));
}
