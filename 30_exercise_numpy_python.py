#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import numpy as np

# Print NumPy version
print("NumPy Version:", np.__version__)

# Print NumPy configuration
print("\nNumPy Configuration:")
print(np.show_config())


# In[3]:


import numpy as np

# Create a null vector of size 10
null_vector = np.zeros(10)

print(null_vector)


# In[4]:


import numpy as np

# Create an example array
my_array = np.zeros((5, 3))

# Find the memory size of the array in bytes
memory_size = my_array.nbytes

print("Memory size of the array:", memory_size, "bytes")


# In[5]:


python -c "import numpy as np; print(np.add.__doc__)"


# In[6]:


import numpy as np

# Create a null vector of size 10
null_vector = np.zeros(10)

# Set the fifth value to 1
null_vector[4] = 1

print(null_vector)


# In[7]:


import numpy as np

# Create a null vector of size 10
null_vector = np.zeros(10)

# Set the fifth value to 1
null_vector[4] = 1

print(null_vector)


# In[8]:


import numpy as np

# Create a vector with values ranging from 10 to 49
my_vector = np.arange(10, 50)

print(my_vector)


# In[9]:


import numpy as np

# Create a vector with values ranging from 10 to 49
my_vector = np.arange(10, 50)

# Reverse the vector
reversed_vector = my_vector[::-1]

print(reversed_vector)


# In[10]:


import numpy as np

# Create a 1D array with values ranging from 0 to 8
my_array = np.arange(9)

# Reshape the 1D array into a 3x3 matrix
my_matrix = my_array.reshape((3, 3))

print(my_matrix)


# In[11]:


import numpy as np

# Create an array
my_array = np.array([1, 2, 0, 0, 4, 0])

# Find indices of non-zero elements
nonzero_indices = np.nonzero(my_array)

print(nonzero_indices)


# In[13]:


import numpy as np

# Create a 3x3 identity matrix
identity_matrix = np.eye(3)

print(identity_matrix)


# In[14]:


import numpy as np

# Create a 3x3x3 array with random values
random_array = np.random.rand(3, 3, 3)

print(random_array)


# In[15]:


import numpy as np

# Create a 10x10 array with random values
random_array = np.random.rand(10, 10)

# Find the minimum and maximum values
min_value = np.min(random_array)
max_value = np.max(random_array)

print("Minimum value:", min_value)
print("Maximum value:", max_value)


# In[16]:


import numpy as np

# Create a 10x10 array with random values
random_array = np.random.rand(10, 10)

# Find the minimum and maximum values
min_value = np.min(random_array)
max_value = np.max(random_array)

print("Minimum value:", min_value)
print("Maximum value:", max_value)


# In[17]:


import numpy as np

# Create a 5x5 array of zeros
array_2d = np.zeros((5, 5))

# Set the border elements to 1
array_2d[0, :] = 1  # Top row
array_2d[-1, :] = 1  # Bottom row
array_2d[:, 0] = 1  # Left column
array_2d[:, -1] = 1  # Right column

print(array_2d)


# In[18]:


import numpy as np

# Expression 1
result1 = 0 * np.nan

# Expression 2
result2 = np.nan == np.nan

# Expression 3
result3 = np.inf > np.nan

# Expression 4
result4 = np.nan - np.nan

# Expression 5
result5 = np.nan in set([np.nan])

# Expression 6
result6 = 0.3 == 3 * 0.1

print("Result of 0 * np.nan:", result1)
print("Result of np.nan == np.nan:", result2)
print("Result of np.inf > np.nan:", result3)
print("Result of np.nan - np.nan:", result4)
print("Result of np.nan in set([np.nan]):", result5)
print("Result of 0.3 == 3 * 0.1:", result6)


# In[19]:


import numpy as np

# Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
diagonal_values = [1, 2, 3, 4]
matrix = np.diag(diagonal_values, k=-1)

print(matrix)


# In[20]:


import numpy as np

# Create a 2x2 checkerboard pattern
checkerboard_tile = np.array([[0, 1], [1, 0]])

# Use tile to repeat the checkerboard pattern to create an 8x8 matrix
checkerboard_matrix = np.tile(checkerboard_tile, (4, 4))

print(checkerboard_matrix)


# In[21]:


import numpy as np

# Define the shape of the array
shape = (6, 7, 8)

# Find the index (x, y, z) of the 100th element
index_100th_element = np.unravel_index(99, shape)

print("Index (x, y, z) of the 100th element:", index_100th_element)


# In[22]:


Index (x, y, z) of the 100th element: (1, 5, 3)


# In[23]:


import numpy as np

# Define the checkerboard pattern
checkerboard_tile = np.array([[0, 1], [1, 0]])

# Use tile to repeat the checkerboard pattern
checkerboard_matrix = np.tile(checkerboard_tile, (4, 4))

print(checkerboard_matrix)


# In[24]:


import numpy as np

# Step 1: Generate a random 5x5 matrix
random_matrix = np.random.rand(5, 5)

# Step 2: Calculate mean and standard deviation
mean_value = np.mean(random_matrix)
std_deviation = np.std(random_matrix)

# Step 3: Normalize the matrix
normalized_matrix = (random_matrix - mean_value) / std_deviation

print("Original Matrix:")
print(random_matrix)

print("\nNormalized Matrix:")
print(normalized_matrix)


# In[25]:


import numpy as np

# Define a custom dtype for RGBA color
rgba_dtype = np.dtype([('R', np.uint8), ('G', np.uint8), ('B', np.uint8), ('A', np.uint8)])

# Create an array with the custom dtype
color_array = np.array([(255, 0, 0, 255),  # Example red color with full opacity
                        (0, 255, 0, 128),  # Example green color with half opacity
                        (0, 0, 255, 0)],   # Example blue color with zero opacity
                       dtype=rgba_dtype)

print(color_array)


# In[26]:


import numpy as np

# Create a 5x3 matrix
matrix_a = np.random.rand(5, 3)

# Create a 3x2 matrix
matrix_b = np.random.rand(3, 2)

# Multiply the matrices
result_matrix = np.dot(matrix_a, matrix_b)
# Alternatively, you can use the @ operator for matrix multiplication
# result_matrix = matrix_a @ matrix_b

print("Matrix A (5x3):")
print(matrix_a)

print("\nMatrix B (3x2):")
print(matrix_b)

print("\nResult Matrix (5x2):")
print(result_matrix)


# In[27]:


import numpy as np

# Create a 1D array
my_array = np.array([1, 4, 6, 3, 9, 2, 7, 5, 8])

# Negate elements between 3 and 8 in place
my_array[(my_array > 3) & (my_array < 8)] *= -1

print(my_array)


# In[28]:


9
10


# In[29]:


import numpy as np

# Assume Z is an integer vector
Z = np.array([1, 2, 3, 4, 5])

# 1. Z**Z
result1 = Z**Z

# 2. 2 << Z >> 2
result2 = 2 << Z >> 2

# 3. Z <- Z
result3 = Z <- Z

# 4. 1j*Z
result4 = 1j*Z

# 5. Z/1/1
result5 = Z/1/1

# 6. Z<Z>Z
# The following line would result in a SyntaxError
# result6 = Z < Z > Z

# Print results
print("1. Z**Z:", result1)
print("2. 2 << Z >> 2:", result2)
print("3. Z <- Z:", result3)
print("4. 1j*Z:", result4)
print("5. Z/1/1:", result5)
# Commenting out the line that causes an error to avoid script termination
# print("6. Z<Z>Z:", result6)


# In[30]:


import numpy as np

# 1. np.array(0) / np.array(0)
result1 = np.array(0) / np.array(0)

# 2. np.array(0) // np.array(0)
result2 = np.array(0) // np.array(0)

# 3. np.array([np.nan]).astype(int).astype(float)
result3 = np.array([np.nan]).astype(int).astype(float)

print("1. np.array(0) / np.array(0):", result1)
print("2. np.array(0) // np.array(0):", result2)
print("3. np.array([np.nan]).astype(int).astype(float):", result3)


# In[31]:


import numpy as np

# Example float array
float_array = np.array([1.25, -2.75, 3.5, -4.8, 5.2])

# Round away from zero
rounded_array = np.where(float_array >= 0, np.ceil(float_array), np.floor(float_array))

print("Original Float Array:", float_array)
print("Rounded Array (away from zero):", rounded_array)


# In[32]:


import numpy as np

# Example arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([3, 4, 5, 6, 7])

# Find common values
common_values = np.intersect1d(array1, array2)

print("Array 1:", array1)
print("Array 2:", array2)
print("Common Values:", common_values)


# In[ ]:




