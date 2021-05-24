# %%
# Source: https://www.w3resource.com/python-exercises/numpy/basic/index.php
import numpy as np

# %%
# Ex 1
print(np.__version__)
print(np.show_config())
# %%
# Ex 2
np.info(np.add)
# %%
# Ex 3
x = np.array([1, 2, 3, 4])
print("Original array:")
print(x)
print("Test if none of the elements of the said array is zero:")
print(np.all(x))

x = np.array([0, 1, 2, 3])
print("\nOriginal array:")
print(x)
print("Test if none of the elements of the said array is zero:")
print(np.all(x))

# %%
# Ex 4
x = np.array([1, 0, 0, 0])
print("Original array:")
print(x)
print("Test whether any of the elements of a given array is non-zero:")
print(np.any(x))

x = np.array([0, 0, 0, 0])
print("\nOriginal array:")
print(x)
print("Test whether any of the elements of a given array is non-zero:")
print(np.any(x))

# %%
# Ex 5
a = np.array([1, 0, np.nan, np.inf])
print("Original array")
print(a)
print("Test a given array element-wise for finiteness:")
print(np.isfinite(a))

# %%
# Ex 6
a = np.array([1, 0, np.nan, np.inf, -np.inf])
print("Original array")
print(a)
print("Test element-wise for positive or negative infinity:")
print(np.isinf(a))

# %%
# Ex 7
a = np.array([1, 0, np.nan, np.inf, -np.inf])
print("Original array")
print(a)
print("Test element-wise for NaN:")
print(np.isnan(a))

# %%
# Ex 8
a = np.array([1 + 1j, 1 + 0j, 4.5, 3, 2, 2j])
print("Original array")
print(a)
print("Checking for complex number:")
print(np.iscomplex(a))
print("Checking for real number:")
print(np.isreal(a))
print("Checking for scalar type:")
print(np.isscalar(3.1))
print(np.isscalar([3.1]))

# %%
