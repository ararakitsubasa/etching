#include <Python.h>
#include <numpy/arrayobject.h>
#include <thread>
#include <vector>

void concatenate_part(PyArrayObject* result, PyArrayObject** arrays, int start, int end, int& current_row) {
    for (int i = start; i < end; ++i) {
        npy_intp* dims = PyArray_DIMS(arrays[i]);
        int rows = dims[0];
        int cols = dims[1];
        
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                *((double*)PyArray_GETPTR2(result, current_row + r, c)) = *((double*)PyArray_GETPTR2(arrays[i], r, c));
            }
        }
        current_row += rows;
    }
}

static PyObject* concatenate(PyObject* self, PyObject* args) {
    PyObject* array_tuple;
    int num_threads;

    if (!PyArg_ParseTuple(args, "Oi", &array_tuple, &num_threads)) {
        return NULL;
    }

    int num_arrays = PyTuple_Size(array_tuple);
    std::vector<PyArrayObject*> arrays(num_arrays);

    int total_rows = 0;
    int cols = 0;

    for (int i = 0; i < num_arrays; ++i) {
        arrays[i] = (PyArrayObject*) PyTuple_GetItem(array_tuple, i);
        npy_intp* dims = PyArray_DIMS(arrays[i]);

        total_rows += dims[0];
        if (i == 0) cols = dims[1];
        else if (cols != dims[1]) {
            PyErr_SetString(PyExc_ValueError, "All input arrays must have the same number of columns.");
            return NULL;
        }
    }

    npy_intp result_dims[2] = { total_rows, cols };
    PyArrayObject* result = (PyArrayObject*) PyArray_SimpleNew(2, result_dims, NPY_DOUBLE);

    int step = num_arrays / num_threads;
    std::vector<std::thread> threads;
    int current_row = 0;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * step;
        int end = (i == num_threads - 1) ? num_arrays : (i + 1) * step;
        threads.emplace_back(concatenate_part, result, arrays.data(), start, end, std::ref(current_row));
    }

    for (auto& t : threads) {
        t.join();
    }

    return PyArray_Return(result);
}

static PyMethodDef ConcatenateMethods[] = {
    {"concatenate", concatenate, METH_VARARGS, "Concatenate multiple numpy arrays along the first axis using multi-threading."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef concatenate_module = {
    PyModuleDef_HEAD_INIT,
    "concatenate_module",
    NULL,
    -1,
    ConcatenateMethods
};

PyMODINIT_FUNC PyInit_concatenate_module(void) {
    import_array();
    return PyModule_Create(&concatenate_module);
}
