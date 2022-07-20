from setuptools import setup
setup(
    name="virga_sniffer",
    version="0.3.4",
    description="Cloud and Virga detection based on radar reflectivity and ceilometer cloud base height.",
    url="https://github.com/jonas-witthuhn/virga_sniffer.git",
    license="GPLv3",
    author="Jonas Witthuhn",
    author_email="remsensarctic@lists.uni-leipzig.de",
    packages=["virga_sniffer"],
    package_dir={"":"src"},
    install_requires=["numpy",
                      "xarray",
                      "netcdf4",
                      "pandas",
                      "bottleneck",
                      "scipy",
                      "matplotlib",
                      ],
    extras_require={
        "example": [
            "jupyter",
            "jupyterlab",
            "ipywidgets",
            "ipympl",
        ],
        "docs": [
            "sphinx",
            "myst-parser"
        ]
    }
)
