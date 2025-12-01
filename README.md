-------------------------
sunspot_detection_size.py
-------------------------

Requires a registered JSOC email address -- insert this near the top of the file (this WILL not work otherwise)

Takes input of an active region (as well as a whole host of optional tuning/datetime parameters), pulls the (S)HARP region that contains that active region, and generates a mask of the umbra/penumbra present within the HARP.  Atttempts to calculate size, and overlays the mask over magnetometer data.

Usage:
Process a current AR:
> python sunspot_detection_size.py 14294 [replace with your region number]

Process a past AR:
> python sunspot_detection_size.py 14274 --t 2025-10-14T01:15:00

--no-size will suppress the size infobox on the final plot  
--ignore-x-left [num] and --ignore-x-right [num] will mask off [num] pixels from the left/right edge -- useful for limb regions

If you're picking up too much non-sunspot stuff, try increasing --bg-sigma (either via param or edit the code itself)  
Default size minimum is 50 pixels (helps avoid picking up odd dark granules and such), but this filters out some tiny umbra/penumbra spots at times
