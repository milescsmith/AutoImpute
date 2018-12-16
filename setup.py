from setuptools import setup


setup(name='AutoImpute',
      version='1.0',
      author='Divyanshu Talwar, Aanchal Mongia, Debarka Sengupta, and Angshul Majumdar',
      author_email="",
      description=("AutoImpute is an auto-encoder based gene-expression (sparse) matrix imputation"),
      packages=["AutoImpute"],
      package_dir={'AutoImpute': 'autoimpute'},
      package_data={'AutoImpute': ['data/*.*','data/raw/*.*']},
      install_requires=['scipy', 'numpy', 'scikit-learn','tensorflow'],
      license="MIT",
      )