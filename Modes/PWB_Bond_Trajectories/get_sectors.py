import h5py
filename = 'reference_bond_nr_0_bend_strength_auto_offset_0.000_no_taper.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
