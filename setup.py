from setuptools import setup, find_namespace_packages
setup(
    name="virga_sniffer",
    version="1.0.0",
    description="Cloud and Virga detection based on radar reflectivity and ceilometer cloud base height.",
    url="https://github.com/remsens-lim/virga_sniffer.git",
    license="GPL-3.0",
    author="Jonas Witthuhn",
    author_email="remsensarctic@uni-leipzig.de",
    #packages=["virga_sniffer"],
    packages=find_namespace_packages(where='src/', include=['virga_sniffer.cmap'])
    package_dir={"": "src"},
    package_data={"": ["*.json"]},
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
