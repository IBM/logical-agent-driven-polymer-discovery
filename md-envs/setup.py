from setuptools import setup, find_packages

setup(name='md-envs',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': [],
    },      
    version='1.0',
    install_requires=['tensorflow<=1.14', 
                      'stable_baselines', 
                      'gym', 
                      'GPy', 
                      'scikit-learn', 
                      'dill',
                      'numpy', 
                      'pandas', 
                      'matplotlib',
                      'tqdm']
)
