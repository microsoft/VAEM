import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers


class vaem_encoders(object):
    def __init__(self,obs_dim,cat_dims,dim_flt,K = 20,latent_dim = 10):

        self._K = K
        self._latent_dim = latent_dim
        self._obs_dim = obs_dim
        self._cat_dims = cat_dims
        self._dim_flt = dim_flt

    def _partial_encoder_local(self,x,mask):
        """
        encoders for marginal VAEs
        """
            
        cumsum_cat_dims = np.concatenate( ([0],np.cumsum(self._cat_dims)))
        dim_cat = len(np.argwhere(self._cat_dims != -1))
        DIM_CAT = (self._cat_dims.sum()).astype(int)
        dim_flt = self._dim_flt
        batch_size = tf.shape(x)[0]
        x_cat = x[:,0:DIM_CAT]
        x_flt = x[:,DIM_CAT:]
        x_flat = tf.cast(tf.reshape(x[:,0:-1], [-1, 1]),tf.float32)
        
        with tf.variable_scope('encoder_local'):
            for d in range(dim_flt-1):
                x_flt_d = x_flt[:,d:d+1]
                local_z = layers.fully_connected(x_flt_d, 50, scope='fc-latent-local-flt-01'+str(d))
                local_z = layers.fully_connected(local_z, 2, scope='fc-latent-local-flt-02'+str(d), activation_fn = None)
                
                if d == 0:
                    local_z_flt_mean = local_z[:,0:1]
                    local_z_flt_logvar = tf.cast(tf.reshape(local_z[:,1],[-1,1]),tf.float32)
                else:
                    local_z_flt_mean = tf.concat([local_z_flt_mean,local_z[:,0:1]],1)
                    local_z_flt_logvar = tf.concat([local_z_flt_logvar,tf.cast(tf.reshape(local_z[:,1],[-1,1]),tf.float32)],1)

            if dim_flt != self._obs_dim:
                for d in range(len(self._cat_dims)):
                    x_cat_d = x_cat[:,cumsum_cat_dims[d]:cumsum_cat_dims[d+1]]
                    local_z = layers.fully_connected(x_cat_d, 50, scope='fc-latent-local-cat-01'+str(d))
                    local_z = layers.fully_connected(local_z, np.asscalar(2*cumsum_cat_dims[d+1]-2*cumsum_cat_dims[d]), scope='fc-latent-local-cat-02'+str(d), activation_fn = None)
                    if d ==0:
                        local_z_cat_mean = local_z[:,0:cumsum_cat_dims[d+1]-cumsum_cat_dims[d]]
                        local_z_cat_logvar = local_z[:,cumsum_cat_dims[d+1]-cumsum_cat_dims[d]:]
                    else:
                        local_z_cat_mean = tf.concat([local_z_cat_mean,local_z[:,0:cumsum_cat_dims[d+1]-cumsum_cat_dims[d]]],1)
                        local_z_cat_logvar = tf.concat([local_z_cat_logvar,local_z[:,cumsum_cat_dims[d+1]-cumsum_cat_dims[d]:]],1)

                encoded_local_mean = tf.concat([local_z_cat_mean,local_z_flt_mean],1)
                encoded_local_logvar = tf.concat([local_z_cat_logvar,local_z_flt_logvar],1)
                encoded_local = tf.concat([encoded_local_mean,encoded_local_logvar],1)
            else:
                encoded_local_mean = local_z_flt_mean
                encoded_local_logvar = local_z_flt_logvar
                encoded_local = tf.concat([encoded_local_mean,encoded_local_logvar],1)
                
        return encoded_local



    def _partial_encoder_global(self,z,x,mask):
        """
        encoders for dependency networks
        """
        batch_size = tf.shape(x)[0]
        z_bias = tf.get_variable("z_bias",shape=[1, 1],initializer=tf.contrib.layers.xavier_initializer())
        z_bias = tf.tile(z_bias, [batch_size, 1])
        z_complete = tf.concat([z,z_bias],1)
        X = tf.concat([x,z_complete],1)
        x_flat = tf.cast(tf.reshape(X, [-1, 1]),tf.float32)
        mask = tf.concat([mask,mask],1)
        
        with tf.variable_scope('encoder_global'):
            F = tf.get_variable("F",shape=[1, self._obs_dim*2, 10],initializer=tf.contrib.layers.xavier_initializer())
            F = tf.tile(F, [batch_size, 1, 1])
            F = tf.reshape(F, [-1, 10])
            b = tf.get_variable("b",shape=[1, self._obs_dim*2, 1],initializer=tf.contrib.layers.xavier_initializer())
            # bias vector
            b = tf.tile(b, [batch_size, 1, 1])
            b = tf.reshape(b, [-1, 1])
            x_aug = tf.concat([x_flat, x_flat * F, b], 1)
            encoded = layers.fully_connected(x_aug, self._K)
            encoded = tf.reshape(encoded,[-1, self._obs_dim*2, self._K])
            mask_on_hidden = tf.cast(tf.reshape(mask,[-1, self._obs_dim*2, 1]),tf.float32)
            mask_on_hidden = tf.tile(mask_on_hidden,[1, 1, self._K])
            encoded = tf.nn.relu(tf.reduce_sum(encoded * mask_on_hidden, 1))
            encoded = layers.fully_connected(encoded, 500)
            encoded = layers.fully_connected(encoded, 200)
            encoded_global = layers.fully_connected(encoded, 2 * (self._latent_dim), activation_fn=None)
            
        return encoded_global
    
    
  
        
        
       