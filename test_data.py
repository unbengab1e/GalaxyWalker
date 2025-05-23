from astropy.table import Table

table = Table.read("/mnt/e/datasets/galaxyWalker/train_add_feature.hdf5")
# print(table.keys())
# print(table.dtype)

print(table.colnames)

table.rename_column('Z_HP', 'redshift')
table.rename_column('sphere_embeddings', 'sphembeddings')
table.rename_column('hyperbolic_embeddings', 'hypembeddings')
table.rename_column('euclidean_embeddings', 'eucembeddings')
table.write("/mnt/e/datasets/galaxyWalker/train_add_feature.hdf5", format='hdf5', path='data', overwrite=True)



