import h5py

path = './results/3dec/linear/run-q-1e-06.hdf5'

with h5py.File(path, 'r+') as f:
    f['parameters'].move('q_0', 'm_q')
    m_q = f['parameters/m_q'][()]
