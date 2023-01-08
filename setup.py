"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import shlex
from warnings import warn

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    readme = f.read()

with open(os.path.join(here, "requirements.txt")) as f:
    install_requires = f.readlines()

with open(os.path.join(here, "sup3r", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")


class PostDevelopCommand(develop):
    """
    Class to run post setup commands
    """

    def run(self):
        """
        Run method that tries to install pre-commit hooks
        """
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            warn("Unable to run 'pre-commit install': {}"
                 .format(e))

        develop.run(self)


setup(
    name="NREL-sup3r",
    version=version,
    description="Super Resolving Renewable Resource Data (sup3r)",
    long_description=readme,
    author="Brandon Benton",
    author_email="brandon.benton@nrel.gov",
    entry_points={
        "console_scripts": ["sup3r=sup3r.cli:main",
                            "sup3r-pipeline=sup3r.pipeline.pipeline_cli:main",
                            "sup3r-batch=sup3r.batch.batch_cli:main",
                            "sup3r-qa=sup3r.qa.qa_cli:main",
                            "sup3r-regrid=sup3r.utilities.regridder_cli:main",
                            "sup3r-visual-qa=sup3r.qa.visual_qa_cli:main",
                            "sup3r-stats=sup3r.qa.stats_cli:main",
                            "sup3r-bias-calc=sup3r.bias.bias_calc_cli:main",
                            "sup3r-solar=sup3r.solar.solar_cli:main",
                            ("sup3r-forward-pass=sup3r.pipeline."
                             "forward_pass_cli:main"),
                            ("sup3r-extract=sup3r.preprocessing."
                             "data_extract_cli:main"),
                            ("sup3r-collect=sup3r.postprocessing."
                             "data_collect_cli:main"),
                            ],
    },
    url="https://github.com/NREL/sup3r",
    packages=find_packages(),
    package_dir={"sup3r": "sup3r"},
    include_package_data=True,
    license="BSD 3-Clause",
    zip_safe=False,
    keywords="sup3r",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        "dev": ["flake8", "pre-commit", "pylint"],
    },
    cmdclass={"develop": PostDevelopCommand},
)
