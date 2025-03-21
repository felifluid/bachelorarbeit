import sys
sys.path.append('./scripts')
import topovis

figs_path = './figs/periodic/'
data_path = './data/circ/'

topovis.main(['-vv', '--triang-method', 'regular', '--plot-out', figs_path+'circ_ns32_fs4_linear.png', '--omit-axes', '--periodic', '--method', 'linear', data_path+'ns32/gkwdata.h5', '1', '4'])