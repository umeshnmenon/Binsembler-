import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='binsembler',
      version='0.1',
      description='Ensembles any number of models predictions based on a chosen metric in the probability bins',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Umesh Narayana Menon',
      packages= setuptools.find_packages(), # ['binsembler'],
      keywords = ['ensembling', 'classifier'],
      zip_safe=False
)