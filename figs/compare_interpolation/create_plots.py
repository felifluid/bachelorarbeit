import sys
sys.path.append('./scripts')
import topovis

figs_path = './figs/compare_interpolation/'
data_path = './data/circ/'

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'circ_ns32_fs1.svg', '--omit-axes', data_path+'ns32/gkwdata.h5', '1', '1'])

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'circ_ns32_fs4.svg', '--omit-axes',data_path+'ns32/gkwdata.h5', '1', '4'])

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'circ_ns128_fs1.svg', '--omit-axes', data_path+'ns128/gkwdata.h5', '1', '1'])