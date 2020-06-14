import setuptools

setuptools.setup(
    name="simple_highway_env",
    version="0.0.1",
    author="Melih Dal",
    author_email="melih.dal@boun.edu.tr",
    description="simple environment for highway behavior",
   packages=setuptools.find_packages(),
    license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
    ),
    install_requires=[
          'numpy', 'gym',
    ]
)