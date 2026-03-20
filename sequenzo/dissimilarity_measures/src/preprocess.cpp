#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <cstring>

namespace py = pybind11;

/*
 * [OPT-11] C++ preprocessing replaces the Python pipeline:
 *
 *   np.unique(seqdata_num, axis=0)       O(n*L*log n) sort
 *   seqconc(seqdata_num)                 O(n*L) Python string construction
 *   seqconc(dseqs_num)                   O(U*L) Python string construction
 *   dict(zip(dseqs_series, range(U)))    O(U) Python dict build
 *   [index_map[e] for e in series]       O(n) Python dict lookups
 *   seqlength(dseqs_num)                 O(U*L) numpy sum
 *
 * With a single O(n*L) C++ pass using word-level FNV-1a hash on raw int rows.
 * No string conversion, no sorting, no Python objects.
 */

namespace {

struct RowHasher {
    size_t ncols;
    explicit RowHasher(size_t ncols) : ncols(ncols) {}
    size_t operator()(const int* row) const {
        // Word-level FNV-1a: hash int32 elements directly.
        // ~4x faster than byte-level on int32 arrays.
        size_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < ncols; ++i) {
            h ^= static_cast<size_t>(static_cast<unsigned int>(row[i]));
            h *= 1099511628211ULL;
        }
        return h;
    }
};

struct RowEqual {
    size_t ncols;
    explicit RowEqual(size_t ncols) : ncols(ncols) {}
    bool operator()(const int* a, const int* b) const {
        return std::memcmp(a, b, ncols * sizeof(int)) == 0;
    }
};

} // anonymous namespace

static std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<int>>
find_unique_sequences(py::array_t<int, py::array::c_style | py::array::forcecast> sequences) {
    const int nseq = static_cast<int>(sequences.shape(0));
    const int ncols = static_cast<int>(sequences.shape(1));
    const int* data = sequences.data();

    std::unordered_map<const int*, int, RowHasher, RowEqual>
        row_map(static_cast<size_t>(nseq), RowHasher(ncols), RowEqual(ncols));

    std::vector<int> didxs(nseq);
    std::vector<int> unique_row_indices;
    unique_row_indices.reserve(static_cast<size_t>(nseq * 0.85 + 64));

    for (int i = 0; i < nseq; ++i) {
        const int* row = data + static_cast<ptrdiff_t>(i) * ncols;
        auto it = row_map.find(row);
        if (it == row_map.end()) {
            int uid = static_cast<int>(unique_row_indices.size());
            row_map[row] = uid;
            unique_row_indices.push_back(i);
            didxs[i] = uid;
        } else {
            didxs[i] = it->second;
        }
    }

    const int nunique = static_cast<int>(unique_row_indices.size());

    // Build unique sequences array
    py::array_t<int> unique_seqs({nunique, ncols});
    int* ubuf = unique_seqs.mutable_data();
    for (int i = 0; i < nunique; ++i) {
        std::memcpy(ubuf + static_cast<ptrdiff_t>(i) * ncols,
                    data + static_cast<ptrdiff_t>(unique_row_indices[i]) * ncols,
                    ncols * sizeof(int));
    }

    // Build didxs output
    py::array_t<int> didxs_arr(nseq);
    std::memcpy(didxs_arr.mutable_data(), didxs.data(), nseq * sizeof(int));

    // Compute sequence lengths: count elements > 0 per row (matches seqlength semantics)
    py::array_t<int> lengths(nunique);
    int* lbuf = lengths.mutable_data();
    for (int i = 0; i < nunique; ++i) {
        const int* row = ubuf + static_cast<ptrdiff_t>(i) * ncols;
        int len = 0;
        for (int j = 0; j < ncols; ++j) {
            if (row[j] > 0) ++len;
        }
        lbuf[i] = len;
    }

    return std::make_tuple(std::move(unique_seqs), std::move(didxs_arr), std::move(lengths));
}
