import setuptools

try:
   import pypandoc
   description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   description = open('README.md').read()

setuptools.setup(
    packages = setuptools.find_packages(),

    install_requires = ['tectosaur'],
    zip_safe = False,
    include_package_data = True,

    name = 'tectosaur_topo',
    version = '0.0.1',
    description = 'Faulting with topography. An app using the tectosaur boundary element library Edit',
    long_description = description,

    url = 'https://github.com/tbenthompson/tectosaur_topo',
    author = 'T. Ben Thompson',
    author_email = 't.ben.thompson@gmail.com',
    license = 'MIT',
    platforms = ['any'],
    classifiers = []
)
