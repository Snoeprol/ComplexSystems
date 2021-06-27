# Effect of objects in predator-prey lattice systems

This library implements the Rosenzweig and MacArthur predator-prey equations with diffusion-reaction dynamics in a square lattice. An circular object or "obstacle" can be added to the lattice.

## Installation

Use git clone to clone the project

This library uses the following packages:

* Numpy
* Matplotlib
* Numba*

\* Numba is used for performance improvements, it is therefore optional

## Usage

```bash
python runs.py
```

Will make the needed data to do the complexity analysis

```python
python rabbit.py
```
Will run the agent-based implementation of the initial model.

# Example results


Super imposed predator and prey latices without object:

![alt text](/Figures/SamplePlots/gif_demo.gif)

Heat map of predator (V) and prey (U) positions in a lattice.:

![alt text](/Figures/SamplePlots/demo_1.png)

Equilibrium phase with object and density dynamics:

![alt text](/Figures/SamplePlots/demo_3.png)

## License
[MIT](https://choosealicense.com/licenses/mit/)
