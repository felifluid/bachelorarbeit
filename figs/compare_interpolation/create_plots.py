import sys
sys.path.append('./scripts')
import topovis

figs_path = './figs/compare_interpolation/'
data_path = './data/circ/'
ext = '.png'

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'circ_ns32/ns32'+ext, '--omit-axes', data_path+'ns32/gkwdata.h5', '1', '1'])

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'circ_ns32/rgi'+ext, '--omit-axes', '--periodic', '--interpolator', 'rgi', data_path+'ns32/gkwdata.h5', '1', '4'])

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'circ_ns32/ns128'+ext, '--omit-axes', data_path+'ns128/gkwdata.h5', '1', '1'])

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'/circ_ns32/rbfi'+ext, '--omit-axes', '--periodic', '--interpolator', 'rbfi', data_path+'ns32/gkwdata.h5', '1', '4'])

data_path = './data/chease/'
figs_path = './figs/compare_interpolation/chease_ns128/'

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'ns128'+ext, '--omit-axes', data_path+'ns128/gkwdata.h5', '1', '1'])

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'fs4_rgi'+ext, '--omit-axes', '--periodic', '--interpolator', 'rgi', data_path+'ns128/gkwdata.h5', '1', '4'])

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'ns512'+ext, '--omit-axes', data_path+'ns512/gkwdata.h5', '1', '1'])

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'fs4_rbfi'+ext, '--omit-axes', '--periodic', '--interpolator', 'rbfi', data_path+'ns128/gkwdata.h5', '1', '4'])