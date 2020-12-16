import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

def data_preprocess(Data,Mask,dic_var_type):
    """
    Convert categorical variables into one hot
    :param Data: data matrix. Note that we assume that the columns of the data matrix is re-ordered, so that the categorical variables appears first, and then the continuous variables afterwards.
    :param Mask: missing mask matrix
    :param dic_var_type: a list that contains the statistical types for each variables
    :return: processed data matrix and mask matrix with one hot encoding representation
    """
    cat_mask = np.zeros(Data.shape[1])
    cat_mask[np.argwhere(dic_var_type == 1)] = 1
    cat_dims = -1 * np.ones(Data.shape[1])
    # number of categories for each variable. If the variable is continuous, than its number of categories is -1.
    for d in range(Data.shape[1]):
        if cat_mask[d] == 1:
            cat_col = Data[:, d].astype('str')
            mask_col = Mask[:, d]
            unique_sym, des_fun = np.unique(cat_col[mask_col != 0], return_inverse=True)
            Data[mask_col != 0, d] = des_fun + 1
            cat_dims[d] = len(unique_sym)

    flt_dims = cat_dims
    flt_dims[np.argwhere(flt_dims == -1)] = 1
    cat_dims = (cat_dims[np.ndarray.flatten(np.argwhere(dic_var_type == 1))]).astype(int)
    flt_len = int(len(np.ndarray.flatten(np.argwhere(dic_var_type == 0)))) # length of continuous variables

    Data_cat = Data[:, np.ndarray.flatten(np.argwhere(dic_var_type == 1))]
    Data_flt = Data[:, np.ndarray.flatten(np.argwhere(dic_var_type == 0))]
    Mask_flt = Mask[:, np.ndarray.flatten(np.argwhere(dic_var_type == 0))]
    Data_cat_oh = np.array([]).reshape(Data.shape[0], 0)
    Mask_cat_oh = np.array([]).reshape(Data.shape[0], 0)

    for d in range(len(cat_dims)):
        
        temp = np.zeros((Data_cat.shape[0], cat_dims[d]))
        temp[Mask[:, d] != 0, :] = cat_to_one_hot(
            Data_cat[Mask[:, d] != 0, d].astype(int), cat_dims[d])
        temp[Mask[:, d] == 0, :] = -1
        Data_cat_oh = np.concatenate([Data_cat_oh, temp], 1)
        temp = np.ones((Data_cat.shape[0], cat_dims[d]))
        temp[Mask[:, d] == 0, :] = 0
        Mask_cat_oh = np.concatenate([Mask_cat_oh, temp], 1)
        
    Data_decompressed = np.concatenate([Data_cat_oh, Data_flt], 1)
    Mask_decompressed = np.concatenate([ Mask_cat_oh, Mask_flt], 1)

    return Data_decompressed, Mask_decompressed, cat_dims, flt_len

def cat_to_one_hot(a, dim_cat):
    
    b = np.zeros((a.shape[0], dim_cat))
    layer_idx = np.arange(a.shape[0]).reshape(a.shape[0])
    b[layer_idx, a - 1] = 1

    return b


def noisy_transform(Data,list_discrete, noise_ratio):
    """
    Add some small noises to continuous-discrete data (defined in Appendix C.1.3 in our paper), i.e numerical data with discrete values.
    Examples include variables that take integer values, for example month, day of week, number of custumors etc.
    Other examples include numerical variables that are recorded on a discrete grid (for example salary).

    :param Data: Data matrix
    :param list_discrete: list of continuous-discrete variables (not categorical variables)
    :param noise_ratio: level of noise divided by the gap between two adjacent discrete values (for example for integer observations, the gap is always one ), maximum value is one
    :return: noisy version of the data matrix, and other parameters that are required in order to remove the noise in the future.
    """
    
    Data_noisy = Data.copy()
    records_d = [] # list of all possible outcomes that are recorded in the continuous-discrete variable
    intervals_d = [] # length of gaps between two adjacent discrete values
    
    for d in range(len(list_discrete)):
        records_d.append(np.unique(Data[:,list_discrete[d]]))
        intervals_d.append(np.concatenate((np.diff(records_d[d]),[0]) )*noise_ratio)
        for m in range(len(records_d[d])):
            pos_value = np.where(Data[:,list_discrete[d]]==records_d[d][m])[0]
            if m ==0:
                noise_l = 0
            else:
                noise_l = -intervals_d[d][m-1]/2

            noise_u = intervals_d[d][m]/2
            noise = np.random.uniform(noise_l,noise_u,len(pos_value))
            Data_noisy[pos_value,list_discrete[d]] += noise
            
    return Data_noisy, records_d, intervals_d



def invert_noise(Data_noisy,list_discrete,records_d):
    """
    Remove the noise added to the continuous-discrete variables.
    (defined in Appendix C.1.3 in our paper), i.e numerical data with discrete values.
    Examples include variables that take integer values, for example month, day of week, number of custumors etc.
    Other examples include numerical variables that are recorded on a discrete grid (for example salary).
    This is normally applied to decoder outputs, so that they are rounded to the closest discrete value,
    as decribed in Appendix C.1.3 in our paper
    :param Data_noisy: noisy data matrix
    :param list_discrete: list of continuous-discrete variables
    :param records_d:
    :return: Data matrix that is rounded to the closest discrete value defined in records_d.
    """
    noise_ratio = 1
    Data_invert = Data_noisy.copy()
    for d in range(len(list_discrete)):
        intervals_d = np.concatenate((np.diff(records_d[d]),[0]) )*noise_ratio
        for m in range(len(records_d[d])):
            if m ==0:
                noise_l = 0
            else:
                noise_l = -intervals_d[m-1]/2

            noise_u = intervals_d[m]/2
            pos_value_noisy = np.intersect1d(np.where(Data_noisy[:,list_discrete[d]]-records_d[d][m]<=noise_u)[0], np.where(records_d[d][m]-Data_noisy[:,list_discrete[d]]<=-noise_l)[0])
            Data_invert[pos_value_noisy,list_discrete[d]] = records_d[d][m]
            
    return Data_invert


def invert_noise_tf(Data_noisy,list_discrete,records_d):
    noise_ratio = 1
    Data_invert = Data_noisy*1
    print(Data_invert)
    for d in range(len(list_discrete)):
        intervals_d = np.concatenate((np.diff(records_d[d]),[0]) )*noise_ratio
        for m in range(len(records_d[d])):
            if m ==0:
                noise_l = 0
            else:
                noise_l = -intervals_d[m-1]

            noise_u = intervals_d[m]/2
            Data_noisy_d = Data_invert[:,list_discrete[d]]
            indicator_1 = tf.sign(Data_noisy_d-records_d[d][m]-noise_u) + tf.sign(records_d[d][m]-Data_noisy_d+noise_l)
            bool_1 = tf.less(indicator_1,-1.5)
            temp_d = Data_noisy_d*(1-tf.cast(bool_1,tf.float32))+ tf.cast(bool_1,tf.float32)*records_d[d][m]
            temp_d = tf.reshape(temp_d,[-1,1])
            temp_less_than_d = Data_invert[:,0:list_discrete[d]]
            temp_greater_than_d = Data_invert[:,list_discrete[d]+1:]
            Data_invert = tf.concat([temp_less_than_d, temp_d, temp_greater_than_d],axis = 1)


    return Data_invert




def compress_data(decoded,cat_dims, dic_var_type,):

    """
    invert the one-hot encoding of categorical variables to discrete values
    :param decoded: output of the VAE decoder
    :param cat_dims: a list containing number of categories of each categorical variables
    :param dic_var_type: a list containing statistical types of each variables.
    :return: data matrix where one hot encodings are removed.
    """


    dim_cat = len(np.argwhere(cat_dims != -1))
    DIM_CAT = (cat_dims.sum()).astype(int)
    decoded_cat = decoded[:,0:DIM_CAT]
    decoded_flt = decoded[:,DIM_CAT:]
    decoded_cat_int = np.zeros((decoded.shape[0],dim_cat))
    cumsum_cat_dims = np.concatenate( ([0],np.cumsum(cat_dims)))

    for d in range(len(cat_dims)):
        decoded_cat_int_p = decoded_cat[:,cumsum_cat_dims[d]:cumsum_cat_dims[d+1]]
        decoded_cat_int_p = decoded_cat_int_p/np.sum(decoded_cat_int_p,1,keepdims=True)
        for n in range(decoded.shape[0]):
            decoded_cat_int[n,d] = np.random.choice(len(decoded_cat_int_p[n,:]), 1 , p=decoded_cat_int_p[n,:])
        print(decoded_cat_int[:,d].max())

    decoded = np.concatenate((decoded_cat_int,decoded_flt),axis=1)

    return decoded

def encode_catrtogrial_column(data, columns):
    le = LabelEncoder()
    for column in columns:
        data[column] = le.fit_transform(data[column])