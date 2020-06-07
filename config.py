# config.py
# simplistic config file to make code platform-agnostic

def getconfig():
    return ConfigMFG()

class ConfigMFG:
    """
    gpu - True means GPU available on plaform , False means it's not; this is used for default values
    os  - 'mac' , 'linux'
    """
    gpu = True
    os = 'linux'




