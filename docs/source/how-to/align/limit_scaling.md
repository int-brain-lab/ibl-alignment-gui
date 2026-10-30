(howto-limit-scaling)=
# How to stop scaling beyond the outermost reference lines

This guide shows you how to keep the regions above the shallowest and below the deepest reference
line at their original size.

With three or more reference line pairs, the regions beyond the outermost lines are scaled by a
straight line fitted through all the pairs. This is usually what you want, but a single outlying
pair can then stretch the whole end of the track.

## Steps

1. Untick `Linear fit` in the top left corner of the Fit figure. The fit is applied again
   straight away.

   ```{image} ../../_static/how_to/align/linear_fit_button.png
   :alt: The Fit figure with Linear fit unticked: the solid fit line leaves the dotted linear fit beyond the outermost reference lines
   :align: center
   ```

The regions beyond the outermost lines are now shifted to meet them, but not scaled. Tick the box
again to go back to the linear fit.

## Related

- {ref}`explanation-how-alignment-works`: what each number of reference lines does
