import numpy as np
import torch
import torch.distributions as tdist
from sklearn.metrics import r2_score
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import pearsonr
from dtaidistance import dtw

def compute_dtw(y, yhat, doprint=False):
    num_outputs = y.shape[1]
    y = y.transpose(1, 0, 2)
    y = y.reshape(num_outputs, -1)
    yhat = yhat.transpose(1, 0, 2)
    yhat = yhat.reshape(num_outputs, -1)
    dtws = np.zeros([num_outputs])
    for i in range(num_outputs):
        dtws[i]  =  dtw.distance(y[i,:], yhat[i,:])

    # print output
    if doprint:
        for i in range(num_outputs):
            print('dtw y{} = {:%.3f}'.format(i + 1, dtws[i]))
    return dtws


def compute_husdfdis(y, yhat, doprint=False):
    num_outputs = y.shape[1]
    y = y.transpose(1, 0, 2)
    y = y.reshape(num_outputs, -1)
    yhat = yhat.transpose(1, 0, 2)
    yhat = yhat.reshape(num_outputs, -1)
    husdfdis = np.zeros([num_outputs])
    for i in range(num_outputs):
        husdfdis[i]  = max(directed_hausdorff(y[i,:], yhat[i,:])[0], directed_hausdorff(yhat[i,:], y[i,:])[0])

    # print output
    if doprint:
        for i in range(num_outputs):
            print('husdfdis y{} = {:%.3f}'.format(i + 1, husdfdis[i]))

    return husdfdis
    
def compute_crcoef(y, yhat, doprint=False):
    num_outputs = y.shape[1]
    y = y.transpose(1, 0, 2)
    y = y.reshape(num_outputs, -1)
    yhat = yhat.transpose(1, 0, 2)
    yhat = yhat.reshape(num_outputs, -1)
    crcoef = np.zeros([num_outputs])
    for i in range(num_outputs):
        crcoef[i], _ = pearsonr(y[i,:], yhat[i,:])

    # print output
    if doprint:
        for i in range(num_outputs):
            print('coefficient y{} = {:%.3f}'.format(i + 1, crcoef[i]))

    return crcoef

def compute_R2(y, yhat, doprint=False):
    num_outputs = y.shape[1]
    y = y.transpose(1, 0, 2)
    y = y.reshape(num_outputs, -1)
    yhat = yhat.transpose(1, 0, 2)
    yhat = yhat.reshape(num_outputs, -1)
    r2 = np.zeros([num_outputs])
    for i in range(num_outputs):
        r2[i] = r2_score(y[i,:], yhat[i,:])

    # print output
    if doprint:
        for i in range(num_outputs):
            print('R2 y{} = {:%.3f}'.format(i + 1, r2[i]))

    return r2
    
    

# computes the VAF (variance accounted for)
def compute_vaf(y, yhat, doprint=False):
    # reshape to ydim x -1
    num_outputs = y.shape[1]
    y = y.transpose(1, 0, 2)
    y = y.reshape(num_outputs, -1)
    yhat = yhat.transpose(1, 0, 2)
    yhat = yhat.reshape(num_outputs, -1)

    diff = y - yhat
    num = np.mean(np.linalg.norm(diff, axis=0) ** 2)
    den = np.mean(np.linalg.norm(y, axis=0) ** 2)
    vaf = 1 - num/den
    vaf = max(0, vaf*100)

    """# new method
    num = 0
    den = 0
    for k in range(y.shape[-1]):
        norm2_1 = (np.linalg.norm(y[:, k] - yhat[:, k])) ** 2
        num = num + norm2_1
        norm2_2 = (np.linalg.norm(y[:, k])) ** 2
        den = den + norm2_2
    vaf = max(0, (1 - num / den) * 100)"""

    # print output
    if doprint:
        print('VAF = {:.3f}%'.format(vaf))

    return vaf


# computes the RMSE for all outputs
def compute_rmse(y, yhat, doprint=False):
    # get sizes from data
    num_outputs = y.shape[1]

    # reshape to ydim x -1
    y = y.transpose(1, 0, 2)
    y = y.reshape(num_outputs, -1)
    yhat = yhat.transpose(1, 0, 2)
    yhat = yhat.reshape(num_outputs, -1)

    rmse = np.zeros([num_outputs])
    for i in range(num_outputs):
        rmse[i] = np.sqrt(((yhat[i, :] - y[i, :]) ** 2).mean())

    # print output
    if doprint:
        for i in range(num_outputs):
            print('RMSE y{} = {:.3f}'.format(i + 1, rmse[i]))

    return rmse

def compute_nrmse(y, yhat, doprint=False):
    # get sizes from data
    num_outputs = y.shape[1]

    # reshape to ydim x -1
    y = y.transpose(1, 0, 2)
    y = y.reshape(num_outputs, -1)
    yhat = yhat.transpose(1, 0, 2)
    yhat = yhat.reshape(num_outputs, -1)

    nrmse = np.zeros([num_outputs])
    for i in range(num_outputs):
        print("y[i, :].shape: ", y[i, :].shape)
        # rms_y = np.sqrt(np.mean(y[i, :] ** 2))
        # print("rms_y",rms_y)
        # # print("np.sqrt(((yhat[i, :] - y[i, :]) ** 2).mean()), rms_y ",np.sqrt(((yhat[i, :] - y[i, :]) ** 2).mean()), rms_y)
        # nrmse[i] = np.sqrt(((yhat[i, :] - y[i, :]) ** 2).mean())/rms_y
        nrmse[i] = np.sqrt(((yhat[i, :] - y[i, :]) ** 2).mean())/(np.std(y[i, :]))
        
        print('x{}: '.format(i + 1), nrmse)
        # print("y rms_y[i]: ", rms_y[i])
    # print output
    if doprint:
        for i in range(num_outputs):
            print('nrmse y{} = {:.3f}'.format(i + 1, nrmse[i]))
            
    print("mean nrmse: ", np.mean(nrmse))
    return nrmse


# computes the marginal likelihood of all outputs
def compute_marginalLikelihood(y, yhat_mu, yhat_sigma, doprint=False):
    # to torch
    y = torch.tensor(y, dtype=torch.double)
    yhat_mu = torch.tensor(yhat_mu, dtype=torch.double)
    yhat_sigma = torch.tensor(yhat_sigma, dtype=torch.double)

    # number of batches
    num_batches = y.shape[0]
    num_points = np.prod(y.shape)
    
    has_zero = (yhat_sigma == 0).any()

    if has_zero:
        print("The tensor contains at least one zero value.")
        return 0
    else:
        print("The tensor does not contain any zero values.")
        
    # get predictive distribution
    pred_dist = tdist.Normal(yhat_mu, yhat_sigma)

    # get marginal likelihood
    marg_likelihood = torch.mean(pred_dist.log_prob(y))
    # to numpy
    marg_likelihood = marg_likelihood.numpy()

    # print output
    if doprint:
        print('Marginal Likelihood / point = {:.3f}'.format(marg_likelihood))

    return marg_likelihood
