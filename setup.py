"""
setup.py
"""
import shlex
from subprocess import check_call
from warnings import warn

from setuptools import setup
from setuptools.command.develop import develop


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
    entry_points={
        "console_scripts": ["sup3r=sup3r.cli:main",
                            "sup3r-pipeline=sup3r.pipeline.pipeline_cli:main",
                            "sup3r-batch=sup3r.batch.batch_cli:main",
                            "sup3r-qa=sup3r.qa.qa_cli:main",
                            "sup3r-bias-calc=sup3r.bias.bias_calc_cli:main",
                            "sup3r-solar=sup3r.solar.solar_cli:main",
                            ("sup3r-forward-pass=sup3r.pipeline."
                             "forward_pass_cli:main"),
                            ("sup3r-collect=sup3r.postprocessing."
                             "data_collect_cli:main"),
                            ],
    },
    package_data={'sup3r': ['postprocessing/writers/*.json']},
    test_suite="tests",
    cmdclass={"develop": PostDevelopCommand},
)
