
<!-- <h1 align="center">
  <br>
  <img src="logo.svg" alt="Redist" width="800">
</h1> -->

![REDIST](logo.svg)

<h3 align="center">A novel reinterpretation method for high-energy physics results.</h4>

## Overview
This is the implementation of a reweighing method for the reinterpretation of binned analyses in high energy physics. The method is based on calculating the change in the distributions of observables, given changes in the decay channel's kinematic distributions.

**Check out the paper [here](https://arxiv.org/pdf/2402.08417.pdf).**

## Installation

You can install `redist` from `pypi`

```bash
pip install redistpy
```

### Manual installation

You can install manually with
```bash
# Clone this repository
git clone https://github.com/lorenzennio/redist.git

# Install
pip install -e redist
```

## Dependencies
This implementation is based on the [pyhf](https://github.com/scikit-hep/pyhf) software for statistical inference. The [examples](examples) use the [EOS](https://github.com/eos/eos) software to calculate theoretical predictions.

### Gradients with JAX (optional)

`redist` runs on any `pyhf` backend. On the JAX backend the likelihood becomes
differentiable, so it can be used with `jax.grad`, `jax.jit` and `jax.vmap`:

```bash
pip install redistpy[jax]
```

```python
import pyhf, jax, jax.numpy as jnp
from redist import modifier

pyhf.set_backend("jax")          # set the backend *before* building the modifier

model = modifier.load("model.json", alt_dist, null_dist)
grad = jax.grad(lambda pars: model.logpdf(pars, data)[0])(pars)
```

The distributions must accept broadcast arrays and be written with operations
JAX can trace: `jax.numpy` rather than `scipy`, and no Python `if` on parameter
values. Adaptive quadrature cannot be traced, so the fallback described below
does not apply here.

Theory codes that are not written in JAX — `EOS`, for example — therefore
cannot be differentiated through. Using them still works on the NumPy backend.

### HAMMER templates (optional)

`redist` normally builds its weights by integrating a null and an alternative
distribution over a kinematic binning. [HAMMER](https://hammer.physics.lbl.gov/)
instead produces templates that are *already* binned in the analysis
observables, reweighted for a choice of form factors and Wilson coefficients, so
the weight in a bin is just the ratio of the two templates there:

```python
from redist import modifier_hammer

reader = modifier_hammer.Reader("config.json")   # cachers -> wrappers -> templates
fitter = reader.createFitter()

cmod = modifier_hammer.Modifier_Hammer(new_pars, alt_dist, null_dist, name="sig")
```

HAMMER is **not** a dependency of `redist` and needs no `pip` flag. It is not on
PyPI — it is a C++ library with a Cython wrapper — so it is imported only when a
`HammerCacher` actually opens a histogram file. Everything else in the module,
including `Modifier_Hammer` itself, works without it.

It is a compiled library called outside the tensor graph, so unlike the rest of
`redist` it cannot be traced: the hammer modifier runs on the NumPy backend only.

To build it (v1.3.0; two patches are needed, one dropping `-ansi` and one
replacing byte literals Cython will not accept as enum values):

```bash
sudo apt-get install -y cmake build-essential libboost-all-dev
git clone --depth 1 --branch v1.3.0 https://gitlab.com/mpapucci/Hammer.git
cd Hammer
sed -i 's/ -ansi -D_FILE_OFFSET_BITS=64/ -D_FILE_OFFSET_BITS=64/' \
  CMakeModules/CompilerChecks.cmake
sed -i -E "s/(UNDEFINED) = b'u'/\1 = 117/; s/(HEADER) = b'b'/\1 = 98/;
           s/(EVENT) = b'e'/\1 = 101/;     s/(HISTOGRAM) = b'h'/\1 = 104/;
           s/(RATE) = b'r'/\1 = 114/;      s/(HISTOGRAM_DEFINITION) = b'd'/\1 = 100/" \
  pyext/wrapper/cppdefs.pxd
cd .. && mkdir Hammer-build && cd Hammer-build
cmake -DCMAKE_INSTALL_PREFIX=../Hammer-install -DWITH_PYTHON=ON \
      -DINSTALL_EXTERNAL_DEPENDENCIES=ON -DFORCE_YAMLCPP_INSTALL=ON ../Hammer
make -j"$(nproc)" && make install
pip install ./pyext
```

The [example](examples/hammer) walks through a B → D\*τν fit. Its two template
`.dat` files are 13 MB and are deliberately not in the repository; the notebook
says where to point it. The same build recipe runs in CI as
`.github/workflows/hammer.yml`, on demand and weekly.

### Bayesian inference (optional)
If you want to perform Bayesian inference with `redist` (or `pyhf`) you'll need to install `bayesian_pyhf`.

You can do so with::
```bash
pip install git+https://github.com/malin-horstmann/bayesian_pyhf.git
```

For visualization of the posterior distribution, `corner` is very useful:

```bash
pip install corner
```

### Bin integrals

The bin integrals use fixed-order Gauss-Legendre quadrature, which calls each
distribution once with one broadcast array per kinematic dimension. A theory
code that can only be evaluated a point at a time — `EOS`, for example — is
detected when the modifier is built and falls back to adaptive quadrature, one
to two orders of magnitude slower. `cmod.quad` reports which rule was picked,
and `quad="gauss"` or `quad="nquad"` forces one.

## Contact

If you come across a bug, have an issue or a question, please file an [issue](https://github.com/lorenzennio/redist/issues/new). For further inquiries, you can talk to us via [Discord](https://discord.gg/bmaVUQcR4w).


## License

MIT
