import os, re, sys
from numpy import get_include
from setuptools import setup, Extension

topdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(topdir, 'MCcubed'))
import VERSION as ver

# C-code source folder
srcdir = os.path.join(topdir, 'src_c', '')
# Include folder with header files
incdir = os.path.join(topdir, 'src_c', 'include', '')

files = os.listdir(srcdir)
# This will filter the results for just the c files:
files = list(filter(lambda x:     re.search('.+[.]c$',     x), files))
files = list(filter(lambda x: not re.search('[.#].+[.]c$', x), files))

inc = [get_include(), incdir]
eca = []
ela = []

extensions = []
for i in range(len(files)):
  print("building '{:s}' extension.".format(files[i].rstrip(".c")))
  e = Extension(files[i].rstrip(".c"),
                sources=["{:s}{:s}".format(srcdir, files[i])],
                include_dirs=inc,
                extra_compile_args=eca,
                extra_link_args=ela)
  extensions.append(e)

setup(name         = "MCcubed",
      version      = "{:d}.{:d}.{:d}".format(ver.MC3_VER, ver.MC3_MIN,
                                             ver.MC3_REV),
      author       = "Patricio Cubillos",
      author_email = "patricio.cubillos@oeaw.ac.at",
      url          = "https://github.com/pcubillos/MCcubed",
      packages     = ["MCcubed"],
      license      = ["MIT"],
      description  = "Multi-Core Markov-Chain Monte Carlo.",
      include_dirs = inc,
      #scripts      = ['MCcubed/mccubed.py'],
      #entry_points={"console_scripts": ['foo = MCcubed.mccubed:main']},
      ext_modules  = extensions)
