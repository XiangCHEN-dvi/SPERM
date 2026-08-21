# Installation

SPERM requires Python 3.10 or newer.

::::{tab-set}
:::{tab-item} pip
```shell
python -m pip install sperm
```
:::
:::{tab-item} uv
```shell
uv add sperm
```
:::
:::{tab-item} Source checkout
```shell
git clone https://github.com/XiangCHEN-dvi/sperm.git
cd sperm
uv sync --extra dev
```
:::
::::

Verify the installation:

```shell
python -c "import sperm; print(sperm.__version__)"
```
