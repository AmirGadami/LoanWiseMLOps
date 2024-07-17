from my_package.configs import config
import os

with open(os.path.join(config.PATH_ROOT,'VERSION')) as f:
    # print(f.read())
    __version__ = f.read().strip()
    print(__version__)