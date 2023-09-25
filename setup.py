import setuptools
import os


lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = ['requests'] # Here we'll add: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    # packages=setuptools.find_packages('py_trsqp', 'py_trsqp.utils', exclude=['test']),
    packages=['py_trsqp', 'py_trsqp.utils'],
    name='py_trsqp',
    version='0.0.3',
    author='Muhammad Iffan Hannanu',
    author_email='iffan.hannanu@gmail.com',
    description='First Official Release',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/iffanh/py-trsqp',
    project_urls = {
    },
    license='GPLv3',
    install_requires=install_requires,
)