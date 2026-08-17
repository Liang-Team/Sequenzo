#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using PamDoubleArray = pybind11::array_t<double, pybind11::array::c_style>;
using PamIntArray = pybind11::array_t<int, pybind11::array::c_style>;

void register_pam_engines(pybind11::module_& module);
