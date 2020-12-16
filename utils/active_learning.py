from __main__ import *
import models.decoders as decoders
import models.encoders as encoders
import models.model as model
import utils.reward as reward

def p_vae_active_learning(Data_train_compressed, Data_train,mask_train,Data_test,mask_test_compressed,mask_test,cat_dims,dim_flt,dic_var_type,args, estimation_method=1):
    
    list_stage = args.list_stage
    list_strategy = args.list_strategy
    epochs = args.epochs
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    p = args.p
    K = args.K
    M = args.M
    Repeat = args.repeat
    iteration = args. iteration
    
    '''
    This function train or loads a VAEM model, and performs SAIA using SING or full EDDI strategy.
    Note that we assume that the last column of x is the target variable of interest
    :param Data_train_compressed: preprocessed traning data matrix without one-hot encodings. Note that we assume that the columns of the data matrix is re-ordered, so that the categorical variables appears first, and then the continuous variables afterwards.
    :param Data_train: preprocessed traning data matrix with one-hot encodings. Is is re-ordered as Data_train_compressed.
    :param mask_train: mask matrix that indicates the missingness of training data,with one-hot encodings. 1=observed, 0 = missing
    :param Data_test: test data matrix, with one-hot encodings.
    :param mask_test: mask matrix that indicates the missingness of test data, with one-hot encodings.. 1=observed, 0 = missing
    :param mask_test_compressed: mask matrix that indicates the missingness of test data, without one-hot encodings. 1=observed, 0 = missing.
    :param cat_dims: a list that indicates the number of potential outcomes for non-continuous variables.
    :param dim_flt: number of continuous variables.
    :param dic_var_type: a list that contains the statistical types for each variables
    :param args.epochs: number of epochs for training.
    :param args.latent_dim: latent dimension of partial VAE.
    :param args.K: dimension of feature map of PNP encoder
    :param args.M: number of samples used for MC sampling
    :param args.Repeat: number of repeats.
    :param estimation_method: what method to use for single ordering information reward estimation.
            In order to calculate the single best ordering, we need to somehow marginalize (average) the
            information reward over the data set (in this case, the test set).
            we provide two methods of marginalization.
            - estimation_method = 0: information reward marginalized using the model distribution p_{vae_model}(x_o).
            - estimation_method = 1: information reward marginalized using the data distribution p_{data}(x_o)
    :param args.list_stage: a list of stages that you wish to perform training. 1 = train marginal VAEs, 2 = train dependency VAEs, 3 = fine-tune, 4 = load a model without training.
    :param args.list_strategy: a list of strategies that is used for SAIA. 0 = Random, 1 = SING
    
    :return: None (active learning results are saved to args.output_dir)
    '''
    n_test = Data_test.shape[0]
    n_train = Data_train.shape[0]
    dim_cat = len(np.argwhere(cat_dims != -1))
    OBS_DIM = dim_cat + dim_flt
    al_steps = OBS_DIM
    # information curves
    information_curve_RAND = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    information_curve_SING = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    rmse_curve_RAND = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    rmse_curve_SING = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    # history of optimal actions
    action_SING = np.zeros((Repeat, n_test,
                            al_steps - 1))
    # history of information reward values
    R_hist_SING = np.zeros(
        (Repeat, al_steps - 1, n_test,
         OBS_DIM - 1))

    for r in range(Repeat):
        ## train partial VAE
        reward_estimation = reward.lindley(M, cat_dims, list_discrete, dic_var_type, records_d,)
        tf.reset_default_graph()
        for stag in range(len(list_stage)):
            stage = list_stage[stag]
            vae = train_p_vae(stage, Data_train, Data_train,mask_train, epochs, latent_dim,cat_dims,dim_flt,batch_size, p, K,iteration)  

        ## Perform active variable selection
        if len(list_strategy)>0:
            for strat in range(len(list_strategy)):
                strategy = list_strategy[strat]
                if strategy == 0:### random strategy
                    ## create arrays to store data and missingness
                    x = Data_test[:, :]  #
    #                 x = np.reshape(x, [n_test, OBS_DIM])
                    mask = np.zeros((n_test, OBS_DIM))
                    mask[:, -1] = 0  # we will never observe target value

                    ## initialize array that stores optimal actions (i_optimal)
                    i_optimal = [ nums for nums in range(OBS_DIM - 1 ) ]
                    i_optimal = np.tile(i_optimal, [n_test, 1])
                    random.shuffle([random.shuffle(c) for c in i_optimal])

                    ## evaluate likelihood and rmse at initial stage (no observation)
                    negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                        x, mask,cat_dims, dic_var_type, M)
                    information_curve_RAND[r, :, 0] = negative_predictive_llh
                    rmse_curve_RAND[r, :, 0] = predictive_rmse
                    for t in range(al_steps - 1 ):
                        print("Repeat = {:.1f}".format(r))
                        print("Strategy = {:.1f}".format(strategy))
                        print("Step = {:.1f}".format(t))
                        io = np.eye(OBS_DIM)[i_optimal[:, t]]
                        mask = mask + io
                        negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                            x, mask, cat_dims, dic_var_type,M)
                        information_curve_RAND[r, :, t +1] = negative_predictive_llh
                        rmse_curve_RAND[r, :, t+1] = predictive_rmse
                    np.savez(os.path.join(args.output_dir, 'UCI_information_curve_RAND.npz'),
                             information_curve=information_curve_RAND)
                    np.savez(os.path.join(args.output_dir, 'UCI_rmse_curve_RAND.npz'),
                             information_curve=rmse_curve_RAND)

                if strategy == 1:### single ordering strategy
                    im_SING = np.zeros((Repeat, al_steps - 1 , M,
                                    n_test, Data_train.shape[1] ))
                    #SING is obtrained by maximize mean information reward for each step for the test set to be consistant with the description in the paper.
                    #We can also get this order by using a subset of training set to obtain the optimal ordering and apply this to the testset.
                    x = Data_test[:, :]  #
#                     x,_, _ = noisy_transform(x, list_discrete, noise_ratio)
    #                 x = np.reshape(x, [n_test, OBS_DIM])
                    mask = np.zeros((n_test, OBS_DIM)) # this stores the mask of missingness (stems from both test data missingness and unselected features during active learing)
                    mask2 = np.zeros((n_test, OBS_DIM)) # this stores the mask indicating that which features has been selected of each data
                    mask[:, -1] = 0  # Note that no matter how you initialize mask, we always keep the target variable (last column) unobserved.
                    negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                        x, mask,cat_dims, dic_var_type, M)
                    information_curve_SING[r, :, 0] = negative_predictive_llh
                    rmse_curve_SING[r, :, 0] = predictive_rmse

                    for t in range(al_steps - 1 ): # t is a indicator of step
                        print("Repeat = {:.1f}".format(r))
                        print("Strategy = {:.1f}".format(strategy))
                        print("Step = {:.1f}".format(t))
                        ## note that for single ordering, there are two rewards.
                        # The first one (R) is calculated based on no observations.
                        # This is used for active learning phase, since single ordering should not depend on observations.
                        # The second one (R_eval) is calculated in the same way as chain rule approximation. This is only used for visualization.
                        if t ==-1:
                            im_0 = Data_train_compressed.reshape((1,Data_train_compressed.shape[0],-1))
                            im = Data_train_compressed.reshape((1,Data_train_compressed.shape[0],-1))
                            R = -1e40 * np.ones((Data_train_compressed.shape[0], OBS_DIM - 1))
                            for u in range(OBS_DIM - 1): # u is the indicator for features. calculate reward function for each feature candidates
                                loc = np.where(mask2[:, u] == 0)[0]
                                if estimation_method == 0:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask,  vae, im_0,loc)
                                else:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask, vae, im, loc)
                        else:
                            R = -1e40 * np.ones((n_test, OBS_DIM - 1))
                            im_0 = reward_estimation.completion(x, mask*0,vae) # sample from model prior
                            im = reward_estimation.completion(x, mask, vae) # sample conditional on observations
                            im_SING[r, t, :, :, :] = im
                            for u in range(OBS_DIM - 1): # u is the indicator for features. calculate reward function for each feature candidates
                                loc = np.where(mask2[:, u] == 0)[0]
                                if estimation_method == 0:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask,  vae, im_0,loc)
                                else:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask, vae, im, loc)
                            R_hist_SING[r, t, :, :] = R
                        
                        i_optimal = (R.mean(axis=0)).argmax() # optimal decision based on reward averaged on all data
                        i_optimal = np.tile(i_optimal, [n_test])
                        io = np.eye(OBS_DIM)[i_optimal]
                        action_SING[r, :, t] = i_optimal
                        mask = mask + io*mask_test_compressed # this mask takes into account both data missingness and missingness of unselected features
                        negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                            x, mask,cat_dims, dic_var_type, M)
                        mask2 = mask2 + io # this mask only stores missingess of unselected features, i.e., which features has been selected of each data
                        information_curve_SING[r, :, t +
                                               1] = negative_predictive_llh
                        rmse_curve_SING[r, :, t+1] = predictive_rmse
                    np.savez(os.path.join(args.output_dir, 'UCI_information_curve_SING.npz'),
                             information_curve=information_curve_SING)
                    np.savez(os.path.join(args.output_dir, 'UCI_rmse_curve_SING.npz'),
                             information_curve=rmse_curve_SING)
                    np.savez(os.path.join(args.output_dir, 'UCI_action_SING.npz'), action=action_SING)
                    np.savez(os.path.join(args.output_dir, 'UCI_R_hist_SING.npz'), R_hist=R_hist_SING)

#             Save results



    return vae

def train_p_vae(stage, x_train, Data_train,mask_train, epochs, latent_dim,cat_dims,dim_flt,batch_size, p, K,iteration):
    '''
        This function trains the partial VAE.
        :param stage: stage of training 
        :param x_train: initial inducing points
        :param Data_train: training Data matrix, N by D
        :param mask_train: mask matrix that indicates the missingness. 1=observed, 0 = missing
        :param epochs: number of epochs of training
        :param LATENT_DIM: latent dimension for partial VAE model
        :param cat_dims: a list that indicates the number of potential outcomes for non-continuous variables.
        :param dim_flt: number of continuous variables.
        :param batch_size: batch_size.
        :param p: dropout rate for creating additional missingness during training
        :param K: dimension of feature map of PNP encoder
        :param iteration: how many mini-batches are used each epoch. set to -1 to run the full epoch.
        :return: trained VAE, together with the test data used for testing.
        '''
    # we have three stages of training.
    # stage 1 = training marginal VAEs, stage 2 = training dependency network, (see Section 2 in our paper)
    # stage 3 = add predictor and improve predictive performance (See Appendix C in our paper)
    if stage ==1:
        load_model = 0
        disc_mode = 'non_disc'
    elif stage == 2:
        load_model = 1
        disc_mode = 'non_local'
    elif stage == 3:
        load_model = 1
        disc_mode = 'joint'
    obs_dim = Data_train.shape[1]
    n_train = Data_train.shape[0]
    list_train = np.arange(n_train)
    batch_size = np.minimum(batch_size,n_train)
    ####### construct
    kwargs = {
        'stage':stage,
        'K': K,
        'latent_dim': latent_dim,
        'batch_size': batch_size,
        'encoder': encoders.vaem_encoders(obs_dim,cat_dims,dim_flt,K, latent_dim ),
        'decoder': decoders.vaem_decoders(obs_dim,cat_dims,list_discrete,records_d),
        'obs_dim': obs_dim,
        'cat_dims': cat_dims,
        'dim_flt': dim_flt,
        'load_model':load_model,
        'decoder_path': os.path.join(args.output_dir, 'generator.tensorflow'),
        'encoder_path': os.path.join(args.output_dir, 'encoder.tensorflow'),
        'x_train':x_train,
        'list_discrete':list_discrete,
        'records_d':records_d,
    }
    vae = model.partial_vaem(**kwargs)
    if iteration == -1:
        n_it = int(np.ceil(n_train / kwargs['batch_size']))
    else:
        n_it = iteration
    hist_loss_full = np.zeros(epochs)
    hist_loss_cat = np.zeros(epochs)
    hist_loss_flt = np.zeros(epochs)
    hist_loss_z_local = np.zeros(epochs)
    hist_loss_kl = np.zeros(epochs)

    if stage == 3:
        ## after stage 2, do discriminative training of y. Please refer to Section 3.4 and Appendix C in our paper.
        for epoch in range(epochs):
            training_loss_full = 0.
            training_loss_cat = 0.
            training_loss_flt = 0.
            training_loss_z_local = 0.
            for it in range(n_it):
                if iteration == -1:
                    batch_indices = list_train[it * kwargs['batch_size']:min(it * kwargs['batch_size'] + kwargs['batch_size'], n_train - 1)]
                else:
                    batch_indices = sample(range(n_train), kwargs['batch_size'])
                x = Data_train[batch_indices, :]
                mask_train_batch = mask_train[batch_indices, :]
                DROPOUT_TRAIN = np.minimum(np.random.rand(mask_train_batch.shape[0], obs_dim), p)
                while True:
                    mask_drop = bernoulli.rvs(1 - DROPOUT_TRAIN)
                    if np.sum(mask_drop > 0):
                        break

                mask_drop = mask_drop.reshape([kwargs['batch_size'], obs_dim])
                _ = vae.update(x, mask_drop * mask_train_batch, 'disc')
                loss_full, loss_cat, loss_flt, loss_z_local, loss_kl, stds, _, _ = vae.full_batch_loss(x,mask_drop * mask_train_batch)
                training_loss_full += loss_full
                training_loss_cat += loss_cat
                training_loss_flt += loss_flt
                training_loss_z_local += loss_z_local

            # average loss over most recent epoch
            training_loss_full /= n_it
            training_loss_cat /= n_it
            training_loss_flt /= n_it
            training_loss_z_local /= n_it
            print(
                'Epoch: {} \tnegative training ELBO per observed feature: {:.2f}, Cat_term: {:.2f}, Flt_term: {:.2f},z_term: {:.2f}'
                    .format(epoch, training_loss_full, training_loss_cat, training_loss_flt, training_loss_z_local))

    for epoch in range(epochs):
        training_loss_full = 0. # full training loss
        training_loss_cat = 0. # reconstruction loss for non_continuous likelihood term
        training_loss_flt = 0. # reconstruction loss for continuous likelihood term
        training_loss_z_local = 0. # reconstruction loss for second stage on z space (gaussian likelihood)
        training_loss_kl = 0 # loss for KL term

        for it in range(n_it):
            if iteration == -1:
                batch_indices = list_train[it*kwargs['batch_size']:min(it*kwargs['batch_size'] + kwargs['batch_size'], n_train - 1)]
            else:
                batch_indices = sample(range(n_train), kwargs['batch_size'])

            x = Data_train[batch_indices, :]
            mask_train_batch = mask_train[batch_indices, :]
            DROPOUT_TRAIN = np.minimum(np.random.rand(mask_train_batch.shape[0], obs_dim), p)
            while True:
                mask_drop = bernoulli.rvs(1 - DROPOUT_TRAIN)
                if np.sum(mask_drop > 0):
                    break

            mask_drop = mask_drop.reshape([kwargs['batch_size'], obs_dim])
            _ = vae.update(x, mask_drop*mask_train_batch,disc_mode)
            loss_full, loss_cat, loss_flt,loss_z_local, loss_kl, stds, _,_ = vae.full_batch_loss(x,mask_drop*mask_train_batch)
            training_loss_full += loss_full
            training_loss_cat += loss_cat
            training_loss_flt += loss_flt
            training_loss_z_local += loss_z_local
            training_loss_kl += loss_kl

          # average loss over most recent epoch
        training_loss_full /= n_it
        training_loss_cat /= n_it
        training_loss_flt /= n_it
        training_loss_z_local /= n_it
        training_loss_kl /= n_it
        hist_loss_full[epoch] = training_loss_full
        hist_loss_cat[epoch] = training_loss_cat
        hist_loss_flt[epoch] = training_loss_flt
        hist_loss_z_local[epoch] = training_loss_z_local
        hist_loss_kl[epoch] = training_loss_kl

        print('Epoch: {} \tnegative training ELBO per observed feature: {:.2f}, Cat_term: {:.2f}, Flt_term: {:.2f},z_term: {:.2f}'
            .format(epoch, training_loss_full,training_loss_cat,training_loss_flt, training_loss_z_local))

    if stage <= 2:
        vae.save_generator(os.path.join(args.output_dir, 'generator.tensorflow'))
        vae.save_encoder(os.path.join(args.output_dir, 'encoder.tensorflow'))

    return vae






