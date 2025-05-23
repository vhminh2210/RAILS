from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Original settings
pyx_directories = ["GraphEnc/evaluator/backend/cpp/", "GraphEnc/util/cython"]
cpp_dirs = ["GraphEnc/evaluator/backend/cpp/include", "GraphEnc/util/cython/include"]

# # Adjustments for Kaggle
# pyx_directories = ["BC-Loss/evaluator/backend/cpp/", "BC-Loss/util/cython"]
# cpp_dirs = ["BC-Loss/evaluator/backend/cpp/include", "BC-Loss/util/cython/include"]

extensions = [
    Extension(
        '*',
        ["*.pyx"],
        extra_compile_args=["-std=c++11"])
]

pwd = os.getcwd()

additional_dirs = [os.path.join(pwd, d) for d in cpp_dirs]

for t_dir in pyx_directories:
    target_dir = os.path.join(pwd, t_dir)
    os.chdir(target_dir)
    ori_files = set(os.listdir("./"))
    setup(
        ext_modules=cythonize(extensions,
                              language="c++"),
        include_dirs=[np.get_include()]+additional_dirs
    )

    new_files = set(os.listdir("./"))
    for n_file in new_files:
        if n_file not in ori_files and n_file.split(".")[-1] in ("c", "cpp"):
            os.remove(n_file)

    os.chdir(pwd)