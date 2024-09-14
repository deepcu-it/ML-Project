from typing import List
from setuptools import find_packages,setup


HYPER_E_DOT ='-e .'
def get_requirements(file_path:str)->List[str]:
    with open('requirements.txt','r') as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","")  for req in requirements]
        if HYPER_E_DOT in requirements:
            requirements.remove(HYPER_E_DOT)
    
    return requirements


setup(
    name='ML-Project',
    version='0.0.1',
    author='Deep',
    author_email='deepghosal445@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)