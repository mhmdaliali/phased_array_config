# phased_array_config

This tool selects the optimal phase-shifter settings for a linear phased-array,
given:

- a measured lookup table for phase shifters (phase vs applied voltage)
- desired beam steering angle
- number of columns (default: 6)

The algorithm chooses the best vector of phase values that achieves the closest possible
inter-column phase progression. It includes:

- robust phase unwrapping
- selection of the best phase vector from the lookup table
- proper fallback for case of large phase margins

### Features
- Fast selection of phase states for each element\Column
- Works directly with real measured lookup tables
- Handles phase discontinuities
- Easy to integrate into a real time beamsteering algorithm

### Usage
```python
from optimizer import compute_optimal_phases
phases = compute_optimal_phases("table.csv", angle_deg=30)
print(phases)

