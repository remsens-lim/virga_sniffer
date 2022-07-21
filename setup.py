from setuptools import setup
setup(
    name="virga_sniffer",
    version="0.3.4",
    description="Cloud and Virga detection based on radar reflectivity and ceilometer cloud base height.",
    url="https://github.com/remsens-lim/virga_sniffer.git",
    license="GPL-3.0",
    author="Jonas Witthuhn",
    author_email="remsensarctic@uni-leipzig.de",
    packages=["virga_sniffer"],
    package_dir={"": "src"},
    package_data={"": ["virga_sniffer/config_*.json"]},
    include_package_data=True,
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
