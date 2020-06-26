from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "ep_checker",
        ["ep_checker.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    name='ep_checker_parallel',
    ext_modules=cythonize(ext_modules, language_level=3, annotate=True)
)
