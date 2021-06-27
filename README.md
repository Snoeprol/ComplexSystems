# Effect of objects in predator-prey lattice systems

This library implements the Rosenzweig and MacArthur predator-prey equations with diffusion-reaction dynamics in a square lattice. An circular object or "obstacle" can be added to the lattice.

## Installation

Use git clone to clone the project

This library uses the following packages:

* Numpy (Tested on 1.19.5)
* Matplotlib (Tested on 3.4.1)
* Numba (Tested on 0.53.1)*

\* Numba is used for performance improvements, it is therefore optional

## Usage
To use this library execute `run.py` with the 4 optional arguments $\beta$, $\gamma$, an object boolean and a video boolean.

The $\beta$ and $\gamma$ parameters refer to the parameters in the transformed version of the Rosenzweig and MacArthur equations. The object boolean will add an obstacle to the lattice. The video boolean will output the results as an MP4, if not `True` the results will be saved as a heat map PNG.

An example run is as follows:

```bash
python runs.py 13 0.5 True False
```
The results will be saved to the /data directory

## Example results

Super imposed predator and prey latices without object animation:

![alt text](/Figures/SamplePlots/gif_demo.gif)

Heat map of predator (V) and prey (U) positions in a lattice.:

![alt text](/Figures/SamplePlots/demo_1.png)

Equilibrium phase with object and density dynamics:

![alt text](/Figures/SamplePlots/demo_3.png)

## License
[MIT](https://choosealicense.com/licenses/mit/)
