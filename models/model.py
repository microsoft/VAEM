import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import re
import copy
import utils.process as process


class partial_vaem(object):
    def __init__(self, stage,encoder,decoder,obs_dim,cat_dims,dim_flt,decoder_path, encoder_path,x_train,list_discrete,records_d,learning_rate=1e-3,optimizer=tf.train.AdamOptimizer,obs_std=0.02*np.sqrt(2),K = 20,latent_dim = 10,batch_size = 100, load_model=0,M=5,all=1):
        
        '''
        :param encoder: type of encoder model choosen from coding.py
        :param decoder: type of decoder model choosen from coding.py
        :param obs_dim: maximum number of partial observational dimensions
        :param cat_dims: a list that indicates the number of potential outcomes for non-continuous variables.
        :param dim_flt: number of continuous variables.
        :param encoder_path: path for saving encoder model parameter
        :param decoder_path: path for saving decoder model parameter
        :param x_train: initial inducing points
        :param list_discrete: list of discrete variables
        :param records_d: unique values of discrete variables
        :param learning_rate: optimizer learning rate
        :param optimizer: we use Adam here.
        :param obs_distrib: Bernoulli or Gaussian.
        :param obs_std: observational noise for decoder.
        :param K: length of code for summarizing partial observations
        :param latent_dim: latent dimension of VAE
        :param batch_size: training batch size
        :param load_model: 1 = load a pre-trained model from decoder_path and encoder_path
        :param M : number of MC samples used when performing imputing/prediction
        '''
        self._stage = stage
        self._K = K
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder
        self._decode = decoder
        self._obs_dim = obs_dim
        self._cat_dims = cat_dims
        self._dim_flt = dim_flt
        self._DIM_CAT = (cat_dims.sum()).astype(int)
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._obs_std = obs_std
        self._load_model = load_model
        self._all = all
        self._encoder_path = encoder_path
        self._decoder_path = decoder_path
        self._M = M
        self._x_train = x_train
        self._list_discrete = list_discrete
        self._records_d = records_d
        self._build_graph()

    #### build VAEM model ####
    def _build_graph(self):
        with tf.variable_scope('is',reuse=tf.AUTO_REUSE):
            # placeholder for training data
            self.x = tf.placeholder(tf.float32, shape=[None, self._obs_dim])
            # placeholder for inducing points
            self.x_induce = tf.placeholder(tf.float32, shape=[self._x_train.shape[0], self._obs_dim])
            # placeholder for observation mask
            self.mask = tf.placeholder(tf.float32, shape=[None, self._obs_dim])
            self._batch_size = tf.shape(self.x)[0]

            ### marginal VAE encoders ###
            # note that the marginal VAEs are stored in the same graph,
            # hence we can train all marginal VAEs simultaneously without the need for additional loop.
            with tf.variable_scope('sampling_local'):
                w = tf.reduce_sum(self.mask)
                self.encoded_local = self._encode._partial_encoder_local(self.x,self.mask)
                # inference net outputs
                self.mean_gaussian_local = self.encoded_local[:, :self._obs_dim-1]
                self.logvar_gaussian_local = self.encoded_local[:, self._obs_dim-1:]
                self.stddev_gaussian_local = tf.sqrt(tf.exp(self.logvar_gaussian_local))

                # Here we have added VampPriors. However, this is only used for visualization purposes when performing data generation.
                # VampPrior will NOT be used when performing data imputation/active learning
                # option I (used in the paper): empirical VampPrior: randomly sample subset of training data to be inducing points
                # at each evaluation, for each data point, a single component of MoG is randomly sampled.
                self.mean_vamp_emp_test_local,self.logvar_vamp_emp_test_local = self._vamp_prior_empirical_local(self._batch_size)

                # option II: optimizable VampPrior
                n_component = 100
                self.mean_vamp_induce_local,self.logvar_vamp_induce_local,self.mean_vamp_induce_batch_local,self.logvar_vamp_induce_batch_local = self._vamp_prior_induce_local(n_component)

                # here we set condition such that, empirical VampPrior is used as prior to generate samples, if no variables are observed (i.e., w = tf.reduce_sum(self.mask)= 0) 
                self.mean_local = tf.cond(tf.equal(w,0), lambda:self.mean_vamp_emp_test_local, lambda: self.mean_gaussian_local)
                self.logvar_local = tf.cond(tf.equal(w,0), lambda:self.logvar_vamp_emp_test_local, lambda: self.logvar_gaussian_local)

                ## final statistics of inference net ##
                self.stddev_local = tf.sqrt(tf.exp(self.logvar_local))
                epsilon_local = tf.random_normal( [self._batch_size, self._obs_dim-1])
                self.z_local = self.mean_local + self.stddev_local * epsilon_local
                
            ### dependency VAE encoders ###
            with tf.variable_scope('sampling_global'):
                # unpacking mean and (diagonal) variance of latent variable
                w = tf.reduce_sum(self.mask)
                self.encoded_global = self._encode._partial_encoder_global(self.z_local,self.x,self.mask)
                self.mean_gaussian_global = self.encoded_global[:, :self._latent_dim]
                self.logvar_gaussian_global = self.encoded_global[:, self._latent_dim:]
                self.stddev_gaussian_global = tf.sqrt(tf.exp(self.logvar_gaussian_global))
                
                # option I: empirical VampPrior: randomly sample subset of trainingdata to be inducing points
                # at each evaluation, for each data point, a single component of MoG is randomly sampled.
                self.mean_vamp_emp_test_global,self.logvar_vamp_emp_test_global = self._vamp_prior_empirical_global(self._batch_size)

                # option II: optimizable VampPrior
                n_component = 100
                self.mean_vamp_induce_global,self.logvar_vamp_induce_global,self.mean_vamp_induce_batch_global,self.logvar_vamp_induce_batch_global = self._vamp_prior_induce_global(n_component)
            
                # here we set condition such that, empirical VampPrior is used as prior to generate samples, if no variables are observed (i.e., w = tf.reduce_sum(self.mask)= 0) 
                self.mean_global = tf.cond(tf.equal(w,0), lambda:self.mean_vamp_emp_test_global, lambda: self.mean_gaussian_global)
                self.logvar_global = tf.cond(tf.equal(w,0), lambda:self.logvar_vamp_emp_test_global, lambda: self.logvar_gaussian_global)

                # final statistics of inference net #
                self.stddev_global = tf.sqrt(tf.exp(self.logvar_global))
                epsilon_global = tf.random_normal([self._batch_size, self._latent_dim])
                self.z_global = self.mean_global + self.stddev_global * epsilon_global
            
            
            ### VAEM decoders ###
            with tf.variable_scope('generator'):
                
                # marginal VAE decoders
                self.auto_std = tf.get_variable("auto_std_local",shape=[1, self._dim_flt],initializer=tf.zeros_initializer()) + self._obs_std
                self.auto_std=tf.stop_gradient(self.auto_std)
                self.auto_std = tf.maximum(self.auto_std,0.0001)
                self.auto_std = tf.minimum(self.auto_std,0.2)
                self.z = tf.concat([self.z_local,self.z_global],1)
                self.decoded_x_from_local, _ = self._decode._vaem_decoder(self.z_local, self.x, self.mask) # x generated from z_local ($\mathcal{z}$ in the paper)
            
            
                # dependency VAE decoder
                self.auto_std_z_local = tf.get_variable("auto_std_global",shape=[1, self._obs_dim-1],initializer=tf.zeros_initializer())+0.01
                self.auto_std_z_local = tf.maximum(self.auto_std_z_local,0.0001)
                self.auto_std_z_local = tf.minimum(self.auto_std_z_local,0.5)
                self.decoded_z_local = layers.fully_connected(self.z_global, 50, scope='fc-01')
                self.decoded_z_local = layers.fully_connected(self.decoded_z_local, 100, scope='fc-02')
                self.decoded_z_local = layers.fully_connected(self.decoded_z_local, self._obs_dim-1, activation_fn=None,scope='fc-final')
                self.decoded_z = self.z_local*self.mask[:,0:-1]+self.decoded_z_local*(1-self.mask[:,0:-1])
                self.decoded_x_from_global, _ = self._decode._vaem_decoder(self.decoded_z, self.x, self.mask,) # x generated from z_global ($\mathcal{h}$ in the paper)
            ### collect variables needed for loss evaluation if stage = 1 ###
            
            if self._stage == 1:
                self.decoded = self.decoded_x_from_local
                self.mean = self.mean_local
                self.stddev = self.stddev_local
                self.logvar = self.logvar_local
                self.mean_vamp_test = self.mean_vamp_emp_test_local
                self.logvar_vamp_test = self.logvar_vamp_emp_test_local
                self.stddev_vamp_test = tf.sqrt(tf.exp(self.logvar_vamp_test))
                self.mean_vamp_induce = self.mean_vamp_induce_local
                self.logvar_vamp_induce = self.logvar_vamp_induce_local
            ### collect variables needed for loss evaluation if stage = 2 ###

            elif self._stage >= 2:
                self.decoded = self.decoded_x_from_global
                self.mean = tf.concat([self.mean_local,self.mean_global],1)
                self.stddev = tf.concat([self.stddev_local,self.stddev_global],1)
                self.logvar = tf.concat([self.logvar_local,self.logvar_global],1)
                self.mean_vamp_test = tf.concat([self.mean_vamp_emp_test_local,self.mean_vamp_emp_test_global],1)
                self.logvar_vamp_test = tf.concat([self.logvar_vamp_emp_test_local,self.logvar_vamp_emp_test_global],1) 
                self.stddev_vamp_test = tf.sqrt(tf.exp(self.logvar_vamp_test))
                self.mean_vamp_induce = tf.concat([self.mean_vamp_induce_local,self.mean_vamp_induce_global],1)
                self.logvar_vamp_induce = tf.concat([self.logvar_vamp_induce_local,self.logvar_vamp_induce_global],1) 
                self.mean_gaussian = tf.concat([self.mean_gaussian_local,self.mean_gaussian_global],1)
                self.stddev_gaussian = tf.concat([self.stddev_gaussian_local,self.stddev_gaussian_global],1)
                
            
            with tf.variable_scope('loss'):
                
                # KL divergence between approximate posterior q and prior p
                with tf.variable_scope('kl-divergence'):
                    self.kl_vamp = self._kl_diagnormal_vamp(self.mean, self.logvar,self.mean_vamp_induce,self.logvar_vamp_induce)
                    self.kl = self._kl_diagnormal_stdnormal( self.mean, self.logvar)
                ### reconstruction loss term for each stages ###
                with tf.variable_scope('gaussian'):

                    if self._stage == 1:
                        self.log_like_z_local = 0
                        self.log_like_z_local_normalized = 0
                        self.log_like_q_z_x = self._gaussian_log_likelihood_pointwise(
                            self.z_local,self.mean,self.stddev) # negative log likelihood of z under inference net distribution
                        self.log_like_p_z_x = self._gaussian_log_likelihood_pointwise(
                            self.z_local,self.mean*0,self.stddev*0+1) # negative log likelihood of z under prior distribution
                        self.log_importance_ratio = self.log_like_p_z_x-self.log_like_q_z_x
                    elif self._stage >=2:
                        self.log_like_z_local = self._gaussian_log_likelihood(
                            self.decoded_z_local,self.z_local,self.auto_std_z_local) # reconstruction loss on z space
                        self.log_like_z_local_normalized = self._gaussian_log_likelihood(
                            self.decoded_z_local/(tf.reduce_max(self.z_local)-tf.reduce_min(self.z_local)),self.z_local/(tf.reduce_max(self.z_local)-tf.reduce_min(self.z_local)),self.auto_std_z_local) # reconstruction loss on z space (normalized loss, for visualization only)
                        self.log_like_q_z_x = self._gaussian_log_likelihood_pointwise(
                            self.z,self.mean,self.stddev) # negative log likelihood of h under inference net
                        self.log_like_p_z_x = self._gaussian_log_likelihood_pointwise(
                            self.z,self.mean*0,self.stddev*0+1) # negative log likelihood of h under prior distribution
                        self.log_importance_ratio = self.log_like_p_z_x-self.log_like_q_z_x # note this is negative ratio

                    self.log_like_flt = self._gaussian_log_likelihood(
                        self.x[:, self._DIM_CAT:] * self.mask[:, self._DIM_CAT:],
                        self.decoded[:, self._DIM_CAT:] * self.mask[:, self._DIM_CAT:],
                        self.auto_std)
                    self.log_like_flt_featurewise = self._gaussian_log_likelihood_featurewise(
                        self.x[:, self._DIM_CAT:] * self.mask[:, self._DIM_CAT:],
                        self.decoded[:, self._DIM_CAT:] * self.mask[:, self._DIM_CAT:],
                        self.auto_std)

                with tf.variable_scope('muti_cat'):
                    if self._dim_flt == self._obs_dim:
                        self.log_like_cat = 0
                        self.decoded_cat_normalized = []
                        self.decoded_flt = self.decoded
                        self.log_like_cat_featurewise = []
                    else:
                        self.log_like_cat,self.decoded_cat_normalized = self._multi_cat_log_likelihood(self.x[:, 0:self._DIM_CAT], self.decoded[:, 0:self._DIM_CAT], self.mask[:, 0:self._DIM_CAT],self._cat_dims)
                        self.log_like_cat_featurewise,_ = self._multi_cat_log_likelihood_featurewise(self.x[:, 0:self._DIM_CAT], self.decoded[:, 0:self._DIM_CAT], self.mask[:, 0:self._DIM_CAT],self._cat_dims)
                        self.decoded_flt = self.decoded[:, self._DIM_CAT:]
                        self.decoded = tf.concat([self.decoded_cat_normalized, self.decoded_flt],1)

                self.log_like = self.log_like_cat + self.log_like_z_local  + self.log_like_flt
                if self._stage ==1:
                    self._beta = 1.0
                elif self._stage ==2:
                    self._beta = 1.0
                else:
                    self._beta = 1.0
                if self._dim_flt == self._obs_dim:
                    self.log_like_featurewise = self.log_like_flt_featurewise
                else:
                    self.log_like_featurewise = tf.concat( [self.log_like_cat_featurewise,self.log_like_flt_featurewise],1)
                    
                ### final loss function ###
                # please note that the final loss actually equals to loss_stage_1 + loss_stage_2. At different stages, we will choose to optimize the corresponding NN parameters of that stage only. Hence in practice, only one loss term can be optimized at the same time.
                self._loss = (self._beta*self.kl+self.log_like) / tf.cast(self._batch_size, tf.float32)  # loss per instance (actual loss used)

                ### display loss decompositions ###
                self._loss_print = (self._beta*self.kl+self.log_like) / tf.reduce_sum(self.mask)  # loss per feature, for tracking training process only
                self._loss_cat_print = (self.log_like_cat) / tf.reduce_sum(self.mask) # reconstruction loss for non_continuous likelihood term
                self._loss_flt_print = (self.log_like_flt) / tf.reduce_sum( self.mask) # reconstruction loss for continuous likelihood term
                self._loss_z_local_print = (self.log_like_z_local_normalized) / tf.reduce_sum(self.mask) # reconstruction loss for second stage on z space (gaussian likelihood)
                self._loss_kl_print = (self.kl) / tf.reduce_sum(
                    self.mask) # loss for KL term
                
            with tf.variable_scope('optimizer'):
                optimizer_joint = self._optimizer(learning_rate=self._learning_rate)
                optimizer_disc = self._optimizer(learning_rate=self._learning_rate)
                optimizer_infer = self._optimizer(learning_rate=self._learning_rate)
                
            disc_variables = []
            non_disc_variables = []
            for v in tf.trainable_variables():
                if "disc" in v.name:
                    disc_variables.append(v)
                else:
                    non_disc_variables.append(v)
                    
            global_variables = []
            non_global_variables = []
            for v in tf.trainable_variables():
                if "global" in v.name:
                    global_variables.append(v)
                else:
                    non_global_variables.append(v)

            local_variables = []
            non_local_variables = []
            for v in tf.trainable_variables():
                if "local" in v.name:
                    local_variables.append(v)
                else:
                    non_local_variables.append(v)

            with tf.variable_scope('training-step'):
                if self._stage == 1:
                    self._train_infer = optimizer_infer.minimize(self._loss,var_list = non_disc_variables)
                elif self._stage ==2:
                    self._train_non_local = optimizer_infer.minimize(self._loss,var_list = non_local_variables)
                elif self._stage ==3:
                    self._train_disc = optimizer_disc.minimize(self._loss, var_list=disc_variables)
                    self._train_joint = optimizer_joint.minimize(self._loss)

            if self._load_model == 1:
                generator_variables = []
                for v in tf.trainable_variables():
                    if "generator" in v.name:
                        generator_variables.append(v)
                encoder_variables = []

                for v in tf.trainable_variables():
                    if "encoder" in v.name:
                        encoder_variables.append(v)

                self._sesh = tf.Session()
                load_encoder = tf.contrib.framework.assign_from_checkpoint_fn(self._encoder_path, encoder_variables)
                load_encoder(self._sesh)
                load_generator = tf.contrib.framework.assign_from_checkpoint_fn(self._decoder_path, generator_variables)
                load_generator(self._sesh)
                uninitialized_vars = []
                
                for var in tf.all_variables():
                    try:
                        self._sesh.run(var)
                    except tf.errors.FailedPreconditionError:
                        uninitialized_vars.append(var)

                init_new_vars_op = tf.variables_initializer(uninitialized_vars)
                self._sesh.run(init_new_vars_op)
            else:
                self._sesh = tf.Session()
                init = tf.global_variables_initializer()
                self._sesh.run(init)
    
    def _vamp_prior_empirical_local(self,n_components):
        idxs = tf.range(tf.shape(self.x_induce)[0])
        ridxs = tf.random_shuffle(idxs)[:n_components]
        x_vamp = tf.gather(self.x_induce, ridxs,axis = 0)
        encoded_vamp = self._encode._partial_encoder_local(x_vamp,x_vamp*0+1)
        mean_vamp = encoded_vamp[:, :self._obs_dim-1]
        logvar_vamp = encoded_vamp[:, self._obs_dim-1:]
        return mean_vamp,logvar_vamp

    def _vamp_prior_induce_local(self,n_components):
        x_vamp = tf.get_variable("x_vamp", shape=[n_components, self._obs_dim], initializer=tf.contrib.layers.xavier_initializer())
        encoded_vamp = self._encode._partial_encoder_local(x_vamp,x_vamp*0+1)
        mean_vamp = encoded_vamp[:, :self._obs_dim-1]
        logvar_vamp = encoded_vamp[:, self._obs_dim-1:]
        ridxs = tf.multinomial(tf.ones([1,n_components],tf.float32), self._batch_size)
        x_vamp_batch = tf.gather(x_vamp, tf.reshape(ridxs,[self._batch_size]),axis = 0)
        encoded_vamp_batch = self._encode._partial_encoder_local(x_vamp_batch,x_vamp_batch*0+1)
        mean_vamp_batch = encoded_vamp_batch[:, :self._obs_dim-1]
        logvar_vamp_batch = encoded_vamp_batch[:, self._obs_dim-1:]
        return mean_vamp,logvar_vamp,mean_vamp_batch,logvar_vamp_batch

    def _vamp_prior_empirical_global(self,n_components):
#             id_vamp = np.random.choice(self._x_train.shape[0], n_components)
        idxs = tf.range(  tf.minimum( tf.minimum(tf.shape(self.x_induce)[0],n_components),tf.shape(self.z_local)[0]))
        ridxs = tf.random_shuffle(idxs)[:n_components]
        z_vamp = tf.gather(self.z_local, ridxs,axis = 0)
        x_vamp = tf.gather(self.x, ridxs,axis = 0)
        encoded_vamp = self._encode._partial_encoder_global(z_vamp,x_vamp,x_vamp*0+1)
        mean_vamp = encoded_vamp[:, :self._latent_dim]
        logvar_vamp = encoded_vamp[:, self._latent_dim:]
        return mean_vamp,logvar_vamp
    
    def _vamp_prior_induce_global(self,n_components):
        z_vamp = tf.get_variable("z_vamp", shape=[n_components, self._obs_dim-1],initializer=tf.contrib.layers.xavier_initializer())
        x_vamp = tf.get_variable("x_vamp",shape=[n_components, self._obs_dim],initializer=tf.contrib.layers.xavier_initializer())
        encoded_vamp = self._encode._partial_encoder_global(z_vamp,x_vamp, x_vamp*0+1)
        mean_vamp = encoded_vamp[:, :self._latent_dim]
        logvar_vamp = encoded_vamp[:, self._latent_dim:]
        ridxs = tf.multinomial(tf.ones([1,n_components],tf.float32), self._batch_size)
        z_vamp_batch = tf.gather(z_vamp, tf.reshape(ridxs,[self._batch_size]),axis = 0)
        x_vamp_batch = tf.gather(x_vamp, tf.reshape(ridxs,[self._batch_size]),axis = 0)
        encoded_vamp_batch = self._encode._partial_encoder_global(z_vamp_batch,x_vamp_batch,x_vamp_batch*0+1)
        mean_vamp_batch = encoded_vamp_batch[:, :self._latent_dim]
        logvar_vamp_batch = encoded_vamp_batch[:, self._latent_dim:]
        return mean_vamp,logvar_vamp,mean_vamp_batch,logvar_vamp_batch

    ### KL divergence ###
    def _kl_diagnormal_stdnormal(self, mu, log_var):
        '''
        This function calculates KL divergence
        :param mu: mean
        :param log_var: log variance
        :return:
        '''
        var = tf.exp(log_var)
#         kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - log_var)
        kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - log_var )
    
        return kl
    
    def _kl_diagnormals_pointwise(self, mu_q, log_var_q,mu_p, log_var_p ):
        '''
        This function calculates KL divergence KL(q//p)
        mu_q & log_var_q: N x D
        mu_p & log_var_p: C x D
        '''
        N = tf.shape(mu_q)[0]
        D = tf.shape(mu_q)[1]
        C = tf.shape(mu_p)[0]
        mu_q = tf.reshape(mu_q,[N,D,1])
        log_var_q = tf.reshape(log_var_q,[N,D,1])
        mu_q = tf.tile(mu_q,[1,1,C])
        log_var_q = tf.tile(log_var_q,[1,1,C])
        mu_p = tf.reshape(tf.transpose(mu_p),[1,D,C])
        log_var_p = tf.reshape(tf.transpose(log_var_p),[1,D,C])
        mu_p = tf.tile(mu_p,[N,1,1])
        log_var_p = tf.tile(log_var_p,[N,1,1])
        var_p = tf.exp(log_var_p)
        var_q = tf.exp(log_var_q)
        kl = 0.5*(log_var_p - log_var_q + (var_q + tf.square(mu_q-mu_p))/var_p -1)
#         kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - log_var)
        return kl

    def _log_normalizing_constant_pointwise(self,mu,log_var):
        z = -0.5*np.log(2*np.pi) - 0.5*np.log(2) - 0.5*log_var
        return z

    def _entropy_diagnormals_pointwise(self,mu,log_var):
        h = 0.5*tf.log(2*np.pi)+0.5*log_var+0.5
        return h

    def _kl_diagnormal_vamp(self, mu, log_var,mu_vamp, log_var_vamp):
        mean_exp_neg_kl = tf.reduce_mean( tf.exp(- self._kl_diagnormals_pointwise(mu,log_var,mu_vamp,log_var_vamp)),axis=2)
        kl = tf.reduce_sum(self._log_normalizing_constant_pointwise(mu,log_var) - tf.log(mean_exp_neg_kl) + self._entropy_diagnormals_pointwise(mu,log_var))
        return kl

    ### likelihood terms
    @staticmethod
    def _bernoulli_log_likelihood(targets, outputs, mask, eps=1e-8):
        '''
        This function comptutes negative log likelihood for Bernoulli likelihood
        :param targets: test data
        :param outputs: model predictions
        :param mask: mask of missingness
        :return: negative log llh
        '''
        eps = 0
        log_like = -tf.reduce_sum(targets * (tf.log(outputs + eps) * mask) + (1. - targets) *  (tf.log((1. - outputs) + eps) * mask))
        return log_like

    @staticmethod
    def _gaussian_log_likelihood(targets, mean, std):
        '''
        This function computes negative log likelihood for Gaussians during training
        Note that constant terms are dropped.
        :param targets: test data
        :param mean: mean
        :param std: sigma
        :return: negative log llh
        '''
        se = tf.reduce_sum(
            0.5 * tf.square(targets[:,0:] - mean[:,0:]) / tf.square(std[:,0:]) + tf.log(std[:,0:]))
        return se
    
    @staticmethod
    def _gaussian_log_likelihood_pointwise(targets, mean, std):
        '''
        This function computes negative log likelihood for Gaussians during training
        Note that constant terms are dropped.
        :param targets: test data
        :param mean: mean
        :param std: sigma
        :return: negative log llh
        '''
        se = tf.reduce_sum(0.5 * tf.square(targets[:,0:] - mean[:,0:]) / tf.square(std[:,0:]) + tf.log(std[:,0:]),axis = 1,keep_dims = True)
        return se
    
    @staticmethod
    def _gaussian_log_likelihood_featurewise(targets, mean, std):
        '''
        This function computes negative log likelihood for Gaussians during training
        Note that constant terms are dropped.
        :param targets: test data
        :param mean: mean
        :param std: sigma
        :return: negative log llh
        '''
        se = 0.5 * tf.square(targets[:,0:] - mean[:,0:]) / tf.square(std[:,0:]) + tf.log(std[:,0:])
        return se

    def _multi_cat_log_likelihood(self, targets, outputs, mask, cat_dims, eps = 1e-2):
        cumsum_cat_dims = tf.cumsum(cat_dims)
        cumsum_outputs = tf.cumsum(outputs,axis = 1)
        local_cumsum_outputs = tf.gather(cumsum_outputs,cumsum_cat_dims-1,axis = 1)
        local_normalizer = tf.concat ([ tf.reshape(local_cumsum_outputs[:,0],[-1,1]), local_cumsum_outputs[:,1:] - local_cumsum_outputs[:,0:-1]],axis = 1)
        local_normalizer = self._tf_repeat(local_normalizer,cat_dims)
        log_like = -tf.reduce_sum(targets * (tf.log(outputs + eps) - tf.log(local_normalizer + eps))*mask)
        decoded_normalized = outputs/local_normalizer
        return log_like, decoded_normalized

    def _multi_cat_log_likelihood_featurewise(self, targets, outputs, mask, cat_dims, eps = 1e-2):
        cumsum_cat_dims = tf.cumsum(cat_dims)
        cumsum_outputs = tf.cumsum(outputs,axis = 1)
        local_cumsum_outputs = tf.gather(cumsum_outputs,cumsum_cat_dims-1,axis = 1)
        local_normalizer = tf.concat ([ tf.reshape(local_cumsum_outputs[:,0],[-1,1]), local_cumsum_outputs[:,1:] - local_cumsum_outputs[:,0:-1]],axis = 1)
        local_normalizer = self._tf_repeat(local_normalizer,cat_dims)
        log_like = -(targets * (tf.log(outputs + eps) - tf.log(local_normalizer + eps))*mask)
        decoded_normalized = outputs/local_normalizer
        return log_like, decoded_normalized


    def _tf_repeat(self,tensor, repeats):
        i = tf.constant(0)
        n = tf.constant(len(repeats))
        op_tensor = tf.reshape(tensor[:, 0], [-1, 1])
        cumsum_repeats = np.cumsum(repeats)
        repeats = tf.constant(repeats)
        cumsum_repeats = tf.constant(cumsum_repeats)
        def condition(op_tensor, tensor, repeats, cumsum_repeats,i, n):
            return i < n
        def body(op_tensor, tensor, repeats, cumsum_repeats, i, n):
            x = tf.tile(tf.reshape(tensor[:, i], [-1, 1]),  tf.cast(tf.reshape([1,repeats[i]],[2]),tf.int32))
            op_tensor = tf.concat([op_tensor, x], axis=1)
            i = i + 1
            return op_tensor, tensor, repeats, cumsum_repeats, i, n
        op_tensor, tensor, repeats, cumsum_repeats, i, n = tf.while_loop(condition, body,[op_tensor, tensor, repeats, cumsum_repeats, i,n], shape_invariants=[tf.TensorShape([tensor.shape[0], None]), tensor.get_shape(), repeats.get_shape(),cumsum_repeats.get_shape(), i.get_shape(), n.get_shape()])
        op_tensor = op_tensor[:, 1:]
        return op_tensor

    ### optimization function
    def update(self, x, mask, mode):
        '''
        This function is used to update the model during training/optimization
        :param x: training data
        :param mask: mask that indicates observed data and missing data locations
        '''
        if mode == 'joint':
            _, loss = self._sesh.run([self._train_joint, self._loss_print],
                                 feed_dict={
                                     self.x: x,
                                     self.mask: mask,
                                     self.x_induce:self._x_train
                                 })
        elif mode == 'disc':
            _, loss = self._sesh.run([self._train_disc, self._loss_print],
                                     feed_dict={
                                         self.x: x,
                                         self.mask: mask,
                                         self.x_induce:self._x_train
                                     })
        elif mode == 'local':
            _, loss = self._sesh.run([self._train_local, self._loss_print],
                                     feed_dict={
                                         self.x: x,
                                         self.mask: mask,
                                         self.x_induce:self._x_train
                                     })
        elif mode == 'non_local':
            _, loss = self._sesh.run([self._train_non_local, self._loss_print],
                                     feed_dict={
                                         self.x: x,
                                         self.mask: mask,
                                         self.x_induce:self._x_train
                                     })
        elif mode == 'non_disc':
            _, loss = self._sesh.run([self._train_infer, self._loss_print],
                                     feed_dict={
                                         self.x: x,
                                         self.mask: mask,
                                         self.x_induce:self._x_train
                                     })
        return loss

    def full_batch_loss(self, x,mask):
        '''
        retrieve different components of loss function
        :param x: dat matrix
        :param mask: mask that indicates observed data and missing data locations
        :return: overall loss (averaged over all entries), KL term, and reconstruction loss
        '''
        loss, loss_cat, loss_flt, loss_z_local, loss_kl,auto_std, log_like_featurewise, log_importance_ratio = self._sesh.run(
            [self._loss_print, self._loss_cat_print, self._loss_flt_print, self._loss_z_local_print, self._loss_kl_print, self.auto_std, self.log_like_featurewise, self.log_importance_ratio],
            feed_dict={
                self.x: x,
                self.mask: mask,
                self.x_induce:self._x_train
            })
        return loss, loss_cat, loss_flt, loss_z_local,loss_kl, auto_std, log_like_featurewise, log_importance_ratio

    ### predictive likelihood and uncertainties
    def predictive_loss(self, x, mask, cat_dims, dic_var_type, M):
        '''
        This function computes predictive losses (negative llh).
        This is used for active learning phase.
        We assume that the last column of x is the target variable of interest
        :param x: data matrix, the last column of x is the target variable of interest
        :param mask: mask that indicates observed data and missing data locations
        :return: MAE and RMSE
        '''
        lh = 0
        rmse = 0
        ae = 0
        uncertainty_data = np.zeros((x.shape[0], M))
        # decompress mask
        mask_flt = mask[:, np.ndarray.flatten(np.argwhere(dic_var_type == 0))]
        mask_cat_oh = np.array([]).reshape(x.shape[0], 0)
        for d in range(len(cat_dims)):
            temp = np.ones((x.shape[0], cat_dims[d]))
            temp[mask[:, d] == 0, :] = 0
            mask_cat_oh = np.concatenate([mask_cat_oh, temp], 1)
        mask = np.concatenate([mask_cat_oh,mask_flt], 1)
        auto_std = self._sesh.run( self.auto_std, feed_dict={ self.x: x,self.mask: mask, self.x_induce:self._x_train})
        for m in range(M):
            decoded_noisy = self._sesh.run(self.decoded, feed_dict={self.x: x,self.mask: mask,self.x_induce:self._x_train})
            decoded = process.invert_noise(decoded_noisy,self._list_discrete,self._records_d)
            target = x[:, -1]
            output = decoded[:, -1]
            uncertainty_data[:, m] = decoded[:, -1]
            lh += np.exp(-0.5 * np.square(target - output) / (
                np.square(auto_std[:,-1])) - np.log(auto_std[:,-1]) - 0.5 * np.log(2 * np.pi))
            rmse += np.sqrt(np.sum(np.square(target - output)) / np.sum(mask.shape[0]))
            ae += np.abs(target - output)
        nllh = -np.log(lh / M)
        rmse /= M
        ae /= M
        return nllh, ae
    
    def get_imputation(self, x, mask_obs, cat_dims, dic_var_type,):
        mask_flt = mask_obs[:, np.ndarray.flatten(np.argwhere(dic_var_type == 0))]
        mask_cat_oh = np.array([]).reshape(x.shape[0], 0)
        for d in range(len(cat_dims)):
            temp = np.ones((x.shape[0], cat_dims[d]))
            temp[mask_obs[:, d] == 0, :] = 0
            mask_cat_oh = np.concatenate([mask_cat_oh, temp], 1)
        mask_obs = np.concatenate([mask_cat_oh,mask_flt], 1)
        decoded_noisy = self._sesh.run(self.decoded,feed_dict={self.x: x, self.mask: mask_obs,self.x_induce:self._x_train})
        z_posterior = self._sesh.run(self.z,feed_dict={self.x: x, self.mask: mask_obs,self.x_induce:self._x_train})
        decoded = process.invert_noise(decoded_noisy,self._list_discrete,self._records_d)
        # revert decode
        dim_cat = len(np.argwhere(cat_dims != -1))
        decoded_cat = decoded[:,0:self._DIM_CAT]
        decoded_flt = decoded[:,self._DIM_CAT:]
        decoded_cat_int = np.zeros((decoded.shape[0],dim_cat))
        cumsum_cat_dims = np.concatenate( ([0],np.cumsum(cat_dims)))
        decoded_cat_p = []
        for d in range(len(cat_dims)):
            decoded_cat_int_p = decoded_cat[:,cumsum_cat_dims[d]:cumsum_cat_dims[d+1]]
            decoded_cat_int_p = decoded_cat_int_p/np.sum(decoded_cat_int_p,1,keepdims=True)
            if d==0:
                decoded_cat_p = decoded_cat_int_p
            else:
                decoded_cat_p = np.concatenate([decoded_cat_p,decoded_cat_int_p],1)
            for n in range(decoded.shape[0]):
                decoded_cat_int[n,d] = np.random.choice(len(decoded_cat_int_p[n,:]), 1 , p=decoded_cat_int_p[n,:])
            print(decoded_cat_int[:,d].max())
        decoded = np.concatenate((decoded_cat_int,decoded_flt),axis=1)
        return decoded,z_posterior,decoded_cat_p

    ### generate partial inference samples
    def im(self, x, mask):
        '''
        This function produces simulations of unobserved variables based on observed ones.
        :param x: data matrix
        :param mask: mask that indicates observed data and missing data locations
        :return: im, which contains samples of completion.
        '''
        im = self._sesh.run(self.decoded,feed_dict={self.x: x, self.mask: mask,self.x_induce:self._x_train})
        return im

    ### calculate the first term of information reward approximation
    def chaini_I(self, x, mask, i,cat_dims, dic_var_type,):
        '''
        calculate the first term of information reward approximation
        used only in active learning phase
        :param x: data
        :param mask: mask of missingness
        :param i: indicates the index of x_i
        :return:  the first term of information reward approximation
        '''
        temp_mask = copy.deepcopy(mask)
        # decompress mask
        temp_mask_flt = temp_mask[:, np.ndarray.flatten(np.argwhere(dic_var_type == 0))]
        temp_mask_cat_oh = np.array([]).reshape(x.shape[0], 0)
        for d in range(len(cat_dims)):
            temp = np.ones((x.shape[0], cat_dims[d]))
            temp[temp_mask[:, d] == 0, :] = 0
            temp_mask_cat_oh = np.concatenate([temp_mask_cat_oh, temp], 1)
        temp_mask_I = np.concatenate([temp_mask_cat_oh, temp_mask_flt ], 1)
        m, v = self._sesh.run([self.mean_gaussian, self.stddev_gaussian],feed_dict={self.x: x, self.mask: temp_mask_I,self.x_induce:self._x_train})
        var = v**2
        log_var = 2 * np.log(v)
        temp_mask[:, i] = 1
        # decompress mask
        temp_mask_flt = temp_mask[:, np.ndarray.flatten(np.argwhere(dic_var_type == 0))]
        temp_mask_cat_oh = np.array([]).reshape(x.shape[0], 0)
        for d in range(len(cat_dims)):
            temp = np.ones((x.shape[0], cat_dims[d]))
            temp[temp_mask[:, d] == 0, :] = 0
            temp_mask_cat_oh = np.concatenate([temp_mask_cat_oh, temp], 1)

        temp_mask_II = np.concatenate([temp_mask_cat_oh, temp_mask_flt, ], 1)
        m_i, v_i = self._sesh.run([self.mean_gaussian, self.stddev_gaussian],feed_dict={self.x: x,self.mask: temp_mask_II,self.x_induce:self._x_train })
        var_i = v_i**2
        log_var_i = 2 * np.log(v_i)
        kl_i = 0.5 * np.sum(np.square(m_i - m) / var + var_i / var - 1. - log_var_i + log_var, axis=1)
        return kl_i

    ### calculate the second term of information reward approximation
    def chaini_II(self, x, mask, i,cat_dims, dic_var_type,):
        '''
        calculate the second term of information reward approximation
        used only in active learning phase
        Note that we assume that the last column of x is the target variable of interest
        :param x: data
        :param mask: mask of missingness
        :param i: indicates the index of x_i
        :return:  the second term of information reward approximation
        '''
        temp_mask = copy.deepcopy(mask)
        temp_mask[:, -1] = 1
        temp_mask_flt = temp_mask[:, np.ndarray.flatten(np.argwhere(dic_var_type == 0))]
        temp_mask_cat_oh = np.array([]).reshape(x.shape[0], 0)
        for d in range(len(cat_dims)):
            temp = np.ones((x.shape[0], cat_dims[d]))
            temp[temp_mask[:, d] == 0, :] = 0
            temp_mask_cat_oh = np.concatenate([temp_mask_cat_oh, temp], 1)

        temp_mask_I = np.concatenate([ temp_mask_cat_oh, temp_mask_flt], 1)
        m, v = self._sesh.run([self.mean_gaussian, self.stddev_gaussian],feed_dict={self.x: x,self.mask: temp_mask_I,self.x_induce:self._x_train})
        var = v**2
        log_var = 2 * np.log(v)
        temp_mask[:, i] = 1
        temp_mask_flt = temp_mask[:, np.ndarray.flatten(np.argwhere(dic_var_type == 0))]
        temp_mask_cat_oh = np.array([]).reshape(x.shape[0], 0)
        for d in range(len(cat_dims)):
            temp = np.ones((x.shape[0], cat_dims[d]))
            temp[temp_mask[:, d] == 0, :] = 0
            temp_mask_cat_oh = np.concatenate([temp_mask_cat_oh, temp], 1)
        temp_mask_II = np.concatenate([temp_mask_cat_oh,temp_mask_flt], 1)
        m_i, v_i = self._sesh.run([self.mean_gaussian, self.stddev_gaussian],feed_dict={ self.x: x,  self.mask: temp_mask_II, self.x_induce:self._x_train })
        var_i = v_i**2
        log_var_i = 2 * np.log(v_i)
        kl_i = 0.5 * np.sum(np.square(m_i - m) / v + var_i / var - 1. - log_var_i + log_var,axis=1)

        return kl_i

    ### save model
    def save_generator(self, path, prefix="is/generator"):
        '''
        This function saves generator parameters to path
        '''
        variables = tf.trainable_variables()
        var_dict = {}
        for v in variables:
            if "generator" in v.name:
                name = prefix + re.sub("is/generator", "", v.name)
                name = re.sub(":0", "", name)
                var_dict[name] = v
        for k, v in var_dict.items():
            print(k)
            print(v)
        saver = tf.train.Saver(var_dict)
        saver.save(self._sesh, path)

    def save_encoder(self, path, prefix="is/encoder"):
        '''
        This function saves encoder parameters to path
        '''
        variables = tf.trainable_variables()
        var_dict = {}
        for v in variables:
            if "encoder" in v.name:
                name = re.sub("is/encoder", "", v.name)
                name = re.sub(":0", "", name)
                var_dict[name] = v
        for k, v in var_dict.items():
            print(k)
            print(v)
        saver = tf.train.Saver(var_dict)
        saver.save(self._sesh, path)
        
        
        
    
    
    



