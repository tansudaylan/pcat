from distutils.core import setup

setup( \
      name='pcat', \
      packages=['pcat'], \
      version='0.1', \
      package_dir={'pcat': 'pcat'}, \
      description='A hierarchical, transdimensional MCMC sampler to explore the catalog space', \
      install_requires=['tdpy'], \
      url='https://github.com/tansudaylan/pcat', \
      author='Tansu Daylan', \
      author_email='tansu.daylan@gmail.com', \
      download_url='https://github.com/tansudaylan/pcat/tarball/v0.1', \
      keywords=['mcmc', 'bayesian', 'transdimensional', 'catalog', 'hierarchical'], \
     )

