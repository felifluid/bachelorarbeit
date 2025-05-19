= Usage <chap:usage>
#set heading(outlined: false)
The ToPoVis code, takes one mandatory parameter, namely the path of the `gkwdata.h5` file, as well as several other optional arguments as input strings.
The script can be called from terminal using the command 

#h(2em) #raw("python topovis.py <args> gkwdata.h5", lang: "sh")

Alternatively, it can also be imported as a local python module from within the same directory using

#h(2em) #raw("import topovis", lang: "py")

and then be called using 

#h(2em) #raw("topovis.main(args: list[str])", lang: "py")

Calling from terminal is preferred for generating a quick visualization.
However, the latter option makes it possible to further analyze and modify the data, or to create embeded plots.
@tab:topovis_args below provides an overview and explanation of all optional arguments.

#show figure: set block(breakable: true)

== List of Arguments
#figure(
  table(
    columns: 4,
    align: left,
    table.header([*argument*], [*description*], [*default*], [*supplement type/values*]),
    [`-v`, `-vv`, `-vvv`], [Causes script to print debugging messages about its progress.], [/], [/],
    [`-h`, `--help`], [Prints a help message.], [/], [/],
    [`--phi`], [Toroidal angle $phi$], [$0.0$], [float],
    [`--poten-timestep`], [Timestep for non-linear potential], [$-1$], [int],
    [`-z`, `--zonal`], [Plot zonal potential. Only used in non-linear simulations.], [/], [/],
    [`--legacy-gmap`], [Calculates the G-factor for $zeta$-shift numerically, instead of using the now standardized `gmap` from `gkwdata.h5`], [/], [/],
    [`--dsf`, `--downsample`], [Downsamples the s-grid by using every nth grid point.], [/], [int],
    [`-p`, `--plot-out`], [Specify a file with extension to which the plot is written to.], [/], [full path, filename],
    [`--triang-method`], [Which triangulation method to use for plotting.], [`regular`], [`'regular', 'delaunay'`],
    [`--plot-grid`], [Plots a combination of scatter and triangulation instead of a contour plot.], [/], [/],
    [`--levels`], [The number of levels to use for the tricontourf plot. Has no effect when combined with `--plot-grid`], [$200$], [int],
    [`--dpi`], [Specifies the dpi of the image plot. Has no effect when plotfile has type 'pdf' or 'svg'.], [$400$], [int],
    [`--omit-axes`], [Hides axis labels and ticks in plot.], [/], [/],
    [`--fx`], [Factor by which to upscale the $psi$-grid.], [/], [int],
    [`--fs`], [Factor by which to upscale the $s$-grid.], [/], [int],
    [`--interpolator`], [Which interpolator to use to interpolate the potential in linear case.], [`rgi`], [`'rgi', 'rbfi'`],
    [`-m`, `--method`], [Method of interpolation.], [`cubic`], [`'nearest', 'linear', 'cubic', 'quintic'`],
    [`-d`, `--data-out`], [Specify a h5-file to which the data is written to.], [/], [full path, filename]
  ),
  caption: [List of all optional ToPoVis arguments.]
) <tab:topovis_args>

== Package Versions
ToPoVis was developed and tested on python 3.13 using the following package versions.

#figure(
  table(
    columns: 5,
    align: left,
    [*package*], [`h5py`], [`matplotlib`], [`numpy`], [`scipy`],
    [*version*], [3.13.0], [3.10.0], [2.2.3], [1.15.2]
  )
)
