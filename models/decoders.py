import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import utils.process as process
class vaem_decoders(object):
    
    def __init__(self,obs_dim,cat_dims, list_discrete,records_d,):

        self._obs_dim = obs_dim
        self._cat_dims = cat_dims
        self._list_discrete = list_discrete
        self._records_d = records_d

    def _vaem_decoder(self, z,X, mask, activation=None):
        z_local = z[:,0:(self._obs_dim-1)]
        cumsum_cat_dims = np.concatenate( ([0],np.cumsum(self._cat_dims)))
        DIM_CAT = (self._cat_dims.sum()).astype(int)
        dim_flt = self._obs_dim - DIM_CAT
        z_cat = z_local[:,0:DIM_CAT]
        z_flt = z_local[:,DIM_CAT:]

        for d in range(len(self._cat_dims)):
            z_cat_d = z_cat[:,cumsum_cat_dims[d]:cumsum_cat_dims[d+1]]
            local_output = layers.fully_connected(z_cat_d, np.asscalar(cumsum_cat_dims[d+1]-cumsum_cat_dims[d]), scope='fc-latent-local-cat-02'+str(d), activation_fn = tf.nn.sigmoid)

            if d ==0:
                local_bias_cat = local_output
            else:
                local_bias_cat = tf.concat([local_bias_cat,local_output],1)

        for d in range(dim_flt-1):
            z_flt_d = z_flt[:,d:d+1]
            local_output = layers.fully_connected(z_flt_d, 1, scope='fc-latent-local-flt-03'+str(d), activation_fn = tf.nn.sigmoid)

            if d == 0:
                local_bias_flt = local_output
            else:
                local_bias_flt = tf.concat([local_bias_flt,local_output],1)

        if len(self._cat_dims)!=0:
            x = tf.concat([local_bias_cat,local_bias_flt],1)
        else:
            x = local_bias_flt

        p = 1.0
        X = p*X[:,0:-1]*mask[:,0:-1] + (1-p)*x*mask[:,0:-1] + x*(1-mask[:,0:-1])
        X = process.invert_noise_tf(X,self._list_discrete,self._records_d)
        Z = tf.concat([X,z],1)
        y = layers.fully_connected(Z, 100, scope='fc-disc-local-01')
        y = layers.fully_connected(y, 100, scope='fc-disc-local-02')
        y = layers.fully_connected(y, 1, activation_fn = tf.nn.sigmoid, scope='fc-disc-local-final')
        X_complete = tf.concat([x,y],1)

        return X_complete, None





