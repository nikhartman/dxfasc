# dxfasc
This module will import `.dxf` files  using `dxfgrabber`, set the electron beam lithography dose based on
emprically determied dose values, choose the order of elements to write, and 
export `.asc` files in the format required by Raith ELPHY Quantum Version 4 software.

### Import

`dxf = dxfgrabber.readfile(filename.dxf)`

### Set Dose

For each polygon in the design the dose is scaled by perimeter/area. The scaling is based on 
empirical evidence for two different writefield sizes. One set of parameters for 120\um 
writefields and another for 1000\um writefields. The writefield size is determined in `get_writefield()`. 
The dose is then calculated in `geometry_to_dose()`.

### Sorting

There are two methods for sorting elements. The simplest `sort_by_position()` will order the polygons
in the design starting at the bottom left corner of the design and moving left to right then 
top to bottom toward the top right corner.

The second method `sort_by_dose()` orders the polygons in the design from highest to lowest dose first,
 then by proximity to the element with the highest dose.
 
 Based on my initial tests, `sort_by_position()` is more reliable when actually writing with the
 electron beam.

### Output

The `.asc` output files are simple ascii files with one text block representing each polygon
in the design. 

```
1 <dose> <layer>
<u1> <v1>
<u2> <v2>
...
<u1> <v1>
#
```

Where (u, v) are the (x, y) cooridinates of the polygon vertices.
