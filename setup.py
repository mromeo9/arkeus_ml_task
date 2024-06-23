from setuptools import find_packages, setup
from typing import List 


E_DOT = '-e .'

def get_requirements(req_file:str) -> List[str]:
    """
    Returns the list of requirements
    """
    
    requirements = []
    with open(req_file) as obj:
        requirements = obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if E_DOT in requirements:
            print('here')
            requirements.remove(E_DOT)
    
    return requirements

setup(
    name="arkeus_ml_task",
    version="0.0.1",
    author="Michael Romeo",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)

