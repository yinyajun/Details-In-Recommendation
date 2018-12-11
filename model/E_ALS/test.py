#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/11 12:16
# @Author  : Yajun Yin
# @Note    :

# import sparse module from SciPy package
from scipy import sparse
# import uniform module to create random numbers
from scipy.stats import uniform
# import NumPy
import numpy as np

# row indices
row_ind = np.array([0, 1, 1, 3, 4])
# column indices
col_ind = np.array([0, 2, 4, 3, 4])
# data to be stored in COO sparse matrix
data = np.array([1, 2, 3, 4, 5], dtype=float)

mat_coo = sparse.coo_matrix((data, (row_ind, col_ind)))
m = mat_coo.tocsc()
print('-------------')
print(m.toarray())
print('-------------')
index = m.indices
ind = m.indptr
data = m.data
print(index)
print(ind)
print(data)
a = m.getrow()