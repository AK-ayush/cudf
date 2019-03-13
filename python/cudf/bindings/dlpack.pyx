# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *

import numpy as np
import pandas as pd
import pyarrow as pa


from libgdf_cffi import ffi, libgdf
from librmm_cffi import librmm as rmm


from libc.stdint cimport uintptr_t, int8_t
from libc.stdlib cimport calloc, malloc, free

cpdef from_dlpack(tensor, num_cols, dtype):

    cdef gdf_size_type* num_columns
    cdef gdf_column** list_cols = <gdf_column**>malloc(num_cols * sizeof(gdf_column*))

    for idx in range(num_cols):
        list_cols[idx] = column_view_from_NDArrays(0, None, mask=None, dtype=dtype, null_count=0)

    cdef gdf_error result

    with nogil:
        result = gdf_from_dlpack(<gdf_column**> list_cols,
                                 <gdf_size_type*> num_columns,
                                 <DLManagedTensor_ const *> &tensor)



