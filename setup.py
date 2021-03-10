from setuptools import setup

setup(name='sudoku_solver',
      version='0.1',
      description='A sudoku solving algorithm with a PyQt5 GUI.',
      url='https://github.com/M-110/sudoku-solver',
      author_email='colinnmclean@gmail.com',
      license='cc0',
      packages=['sudoku_solver'],
      install_requires=['PyQt5'],
      entry_points={'console_scripts': ['sudoku-solver=sudoku_solver.__main__:main']})
