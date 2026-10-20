# Units and differentiation

- Cell lengths and wavelength are in angstroms. Cell angles and profile grids
  are in degrees. The grid is the scattering angle 2theta.
- Model `q_min` and `q_max` use q = 4 pi sin(theta) / wavelength. They bound
  reflection enumeration, not the output grid.
- TCH/FCJ Gaussian variance parameters use centidegrees squared and Lorentzian
  width parameters use centidegrees. Do not pass widths in degrees into those
  instrument dictionaries without conversion.
- Profile histograms are signal-only. Background is supplied separately or
  eliminated with scale by the weighted profile objective.
- The profile objective returns a loss for each pattern. Reduce that vector
  explicitly (for example, with `.mean()`) when fitting one shared model to
  several patterns.
- Request `dtype=torch.float64` when creating structure batches. The paper's
  numerical validation uses double precision. The API default remains float32.
- `compile_reflections` and `forward_with_layers_fixed_hkls` keep a reflection
  list fixed during local refinement. Choose the initial enumeration range to
  cover the intended parameter variation. Large structural changes may require
  a new list; enumeration itself is not differentiated.
- Reflection membership, peak grouping, and finite profile support are discrete
  choices. Gradients apply within the resulting piecewise-smooth calculation.
- This release concerns constant-wavelength X-ray diffraction. It does not
  implement time-of-flight, magnetic scattering, or stacking-fault models.

The examples use broad synthetic peaks, small cell changes, and float64. Their
purpose is to make the calculation and differentiation easy to inspect, not to
set universal optimization defaults for experimental data.
