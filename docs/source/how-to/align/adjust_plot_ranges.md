(howto-adjust-plot-ranges)=
# How to change the colour and axis ranges of the plots

This guide shows you how to bring out a feature that is washed out or saturated at the default
range.

## Steps

1. Open `Plugins → Range Controller`.

   ```{image} ../../_static/how_to/align/range_controller.png
   :alt: The range controller window, with minimum and maximum sliders and values for the image, line and probe plots
   :align: center
   ```

2. Drag the minimum and maximum sliders of the plot type, or type exact values in the boxes beside
   them:

   - image and probe plots: the colour scale
   - line plots: the x axis

3. With more than one shank or configuration, tick the shanks (or **All**) and configurations (or
   **both**) the ranges should apply to.
4. Click **Reset** beside a plot type to return it to its default range, or press {kbd}`R` to reset
   every plot.

## Related

- {ref}`ref-plugins`: range controller
