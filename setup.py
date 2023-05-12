import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='py-trsqp',
    version='0.0.1',
    author='Muhammad Iffan Hannanu',
    author_email='iffan.hannanu@gmail.com',
    description='Testing',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/iffanh/py-trsqp',
    project_urls = {
    },
    license='GNU',
    packages=['py-trsqp'],
    install_requires=['requests'],
)