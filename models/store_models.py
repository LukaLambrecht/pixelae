# import models and store them as keras files for easier loading

import os
import sys

# make output directory
modeldir = 'models'
if not os.path.exists(modeldir): os.makedirs(modeldir)

# models for cluster position in xy coordinates in pixel disks
import modeldefs_clusterposition
model = modeldefs_clusterposition.model_clusterposition_pxdisks_mwe((32, 32, 1))
model.save(os.path.join(modeldir,'model_clusterposition_pxdisks_mwe.keras'))

# models for cluster occupancy for pixel layers and rings
import modeldefs_clusters
model = modeldefs_clusters.model_clusters_pxlayers_test((64, 24, 1))
model.save(os.path.join(modeldir,'model_clusters_pxlayers_test.keras'))
model = modeldefs_clusters.model_clusters_pxrings_test((48, 88, 1))
model.save(os.path.join(modeldir,'model_clusters_pxrings_test.keras'))
