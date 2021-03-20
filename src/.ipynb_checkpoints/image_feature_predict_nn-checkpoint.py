import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import os
import tifffile as tfl
import matplotlib.pyplot as plt


# build neural network to predict image feature vectors based on response vectors
class ImageFeaturePredict:
    
    def __init__(self,resp_vectors,image_feature_vectors):
        
        self.resp = resp_vectors # shape of (nCells,nFrames)
        self.feat = image_feature_vectors # shape of (nFrames,nWavelets)
    
    class Network(nn.Module):
        
        def __init__(self,n_cells,n_wavelets,hidden_dim_list,dropout_p):
            '''initialize constructed model with input layer, output layer, 
            and len(hidden_dim_list) hidden layers
            input_layer size is number of cells in response vector
            output_layer size is number of image features in feature vector'''
        
            # calling the constructor of the parent class - gets everything we need from pytorch
            super(ImageFeaturePredict.Network, self).__init__()
            
            ## Define layers of network
            # first layer input: # of cells in a response_vector
            self.input_layer = nn.Linear(n_cells,hidden_dim_list[0])
            
            # add hidden layers of network based on list of dimensions for each hidden layer
            self.hidden_layers = [] * len(hidden_dim_list)
            for i in range(len(hidden_dim_list) - 1):
                self.hidden_layers.append(nn.Linear(hidden_dim_list[i],hidden_dim_list[i+1]))
            
            # last layer output: # of Gabor wavelets/image features in a feature_vector
            self.output_layer = nn.Linear(hidden_dim_list[-1],n_wavelets)
            
            # dropout
            self.dropout = nn.Dropout(p=dropout_p)
            
            # non-linear activation function between layers
            self.relu = nn.ReLU() # use relu to transform activiations between hidden layers
            self.tanh = nn.Tanh() # use tanh to keep activations between -1 and 1 at final layer
            
        
        def forward(self,batch):
            '''define the forward steps in the model between the layers;
            data batches are passed between layers after computing the non-linear transformations 
            on the activations'''
            
            # input layer
            batch = self.input_layer(batch)
            batch = self.relu(batch)
      
            # hidden layers
            for h in self.hidden_layers:
                batch = h(batch)
                batch = self.dropout(batch)
                batch = self.relu(batch)
            
            # output layer
            batch = self.output_layer(batch)
            batch = torch.tanh(batch)
            
            return batch
        

    def train_test_network(self,n_folds,test_size,n_epochs,batch_size,hidden_dim_list,dropout_p,learning_rate,save_path):
        '''train model using specified parameters, then test model accuracy (using MSE)
        :param n_folds: (int) the number of folds used for cross-validation; if 1, then use test_size to determine split
        :param test_size: (float) the fraction of data to use as as test data if n_folds is 1
        :param n_epochs: (int) number of epochs per training period
        :param batch_size: (int) size of data batches to learn on during each epoch
        :param hidden_dim_list: (list) list specifying the dimension of each hidden layer in the model, len(hidden_dim_list) == number of hidden layers
        :param dropout_p: (float) decimal specifying the dropout probability of nodes in hidden layers during learning steps
        :param learning_rate: (float) rate/step size of descent during learning
        :param save_path: (string) path to save model
        '''
        
        # define loss function
        loss_function = nn.MSELoss()
        
        # define k-fold cross validation parameters
        fold_avg_mse = {}
        fold_mses = []
        fold_test_avg_mse = []
        
        # if using 1 fold, use test_size to determine size of single split, else split based on number of folds
        if n_folds == 1:
            splits = [(None,None)]
        else:
            kfold = KFold(n_splits=n_folds,random_state=42,shuffle=True)
            splits = kfold.split(self.resp)
        
        print('-----------------------')
        # iterate over folds
        for fold,(train_ids, test_ids) in enumerate(splits):
    
            print('Fold ',fold+1)
            print('-----------------------')
        
            ## organize data
            # split the data into a training and test set
            # if using 1 fold, use test_size to determine size of single split, else split based on number of folds
            if n_folds == 1:
                train_resp,test_resp,train_feat,test_feat = train_test_split(self.resp,self.feat,test_size=test_size,random_state=42,shuffle=True)
            else:
                train_resp = self.resp[train_ids]
                train_feat = self.feat[train_ids]
                test_resp = self.resp[test_ids]
                test_feat = self.feat[test_ids]
            
            # create data batches
            train_resp_batches,train_feat_batches = batch_data(train_resp,train_feat,batch_size=batch_size)

            ## organize model and learning conditions
            # construct network
            neural_network = ImageFeaturePredict.Network(len(train_resp[0]),len(train_feat[0]),hidden_dim_list,dropout_p)

            # define optimization function on network - using Adam
            if learning_rate is not None:
                optimizer = optim.Adam(neural_network.parameters(),lr=learning_rate)
            else:
                optimizer = optim.Adam(neural_network.parameters())

            ## begin to train the network
            neural_network.train()

            # train model over specified number of epochs
            epoch_mses = []
            for iE in range(n_epochs):

                # keep track of performance (using mse) over multiple batches of data
                batch_mses = []

                # iterate over each data batch
                for iB in range(len(train_resp_batches)):

                    # pull out data batch (resp=data batch) (feat=labels)
                    resp = train_resp_batches[iB]
                    feat = train_feat_batches[iB]

                    # reset optimizer
                    optimizer.zero_grad()

                    # calculate model predictions on batch
                    predictions = neural_network(torch.tensor(resp.astype(np.float32)))

                    # calculate error of prediction
                    loss = loss_function(predictions,torch.FloatTensor(feat))
                    # backward calculation step
                    loss.backward()

                    # calculate and update weights of network
                    optimizer.step()

                    # test model predictions at this epoch using MSE as measure of accuracy
                    for iF,prediction in enumerate(predictions.data):
                        real_fv = feat[iF]
                        pred_fv = np.asarray(prediction)

                        mse = np.square(np.subtract(real_fv,pred_fv)).mean()
                        batch_mses.append(mse)

                # report prediction performance       
                print('Average MSE for Epoch #'+str(iE+1)+': '+str(np.mean(batch_mses)))
                epoch_mses.append(np.mean(batch_mses))
                
            # record MSE across all epochs for each fold
            fold_mses.append(epoch_mses)
            
            ## evaluate the model performance on the fold using test data
            with torch.no_grad():
                
                # calculate predictions on test data
                test_predictions = neural_network(torch.tensor(test_resp.astype(np.float32)))

                # test model predictions on test data using MSE as measure of accuracy
                test_mses = []
                for iF,prediction in enumerate(test_predictions.data):
                    real_fv = test_feat[iF]
                    pred_fv = np.asarray(prediction)

                    mse = np.square(np.subtract(real_fv,pred_fv)).mean()
                    test_mses.append(mse)

                # report prediction performance       
                print('Average MSE of test data: ' + str(np.mean(test_mses)) )
                print('-----------------------')
                
                fold_test_avg_mse.append(np.mean(test_mses))

        # end training over folds and return the final fold's test data for further evaluation
        neural_network.eval()
        if n_folds != 1:
            avg_mse_folds = sum(fold_test_avg_mse)/len(fold_test_avg_mse)
            print('Average MSE of test data across all folds: ',avg_mse_folds)

        return neural_network,test_resp,test_feat,fold_mses,fold_test_avg_mse
    
    
    def test_predicted_image_correlations(self,model,in_resp,in_feat,im,G_rev):
        '''test pearson's correlation between the image reconstructed from the
        predicted feature vector and (1) the reconstructed image from the ground
        truth feature vector and (2) the original image'''
        r_recon_pred_list = []
        r_orig_pred_list = []
        all_feat = self.feat

        # loop through each response to predict
        for iF in range(in_resp.shape[0]):
            # calculate the model-predicted feature vector based on the response
            pred_fv = np.asarray(model(torch.tensor(in_resp[iF,:].astype(np.float32))).data)
            # calculate the predicted image
            pred_im = features2image(pred_fv,G_rev)

            # create the image reconstructed from the ground truth feature vector
            recon_im = features2image(in_feat[iF,:],G_rev)

            # calculate the correlation b/w pred_im and recon_im
            r_recon_pred,_ = pearsonr(recon_im.flatten(),pred_im.flatten())
            r_recon_pred_list.append(r_recon_pred)

            # find the index of the original image using the ground truth feature vector
            orig_im_idx = np.where(np.all(all_feat[:im.shape[2],:]==in_feat[iF],axis=1))[0][0]
            orig_im = im[:,:,orig_im_idx]

            # calculate the correlation b/w pred_im and orig_im
            r_orig_pred,_ = pearsonr(orig_im.flatten(),pred_im.flatten())
            r_orig_pred_list.append(r_orig_pred)

        r_recon_pred_arr = np.asarray(r_recon_pred_list)
        r_orig_pred_arr = np.asarray(r_orig_pred_list)

        # report the mean correlations
        print('Avg Corr (r) - recon and pred: ',r_recon_pred_arr.mean())
        print('Avg Corr (r) - orig and pred: ',r_orig_pred_arr.mean())

        return r_recon_pred_arr,r_orig_pred_arr
    
    
    def save_predicted_movie(self,model,resp,G_rev,image_size,save_path):
        '''calculate predicted feature vectors, reconstruct images, and save into .tif file'''
        n_frames = resp.shape[0]
        pred_im = np.zeros((n_frames,image_size[0],image_size[1]))
        for iF in range(n_frames):
            pred_fv = np.asarray(model(torch.tensor(resp[iF,:].astype(np.float32))).data)
            pred_im[iF,:,:] = features2image(pred_fv,G_rev)
        save_name = os.path.join(save_path,'predicted_movie.tif')
        tfl.imsave(save_name,pred_im,bigtiff=True)

        return pred_im
    
    
### utility fxns
def batch_data(data,labels,batch_size=16):
    '''batch data for model training'''
    data_batches = []
    label_batches = []
    ids = []

    for n in range(0,len(data),batch_size):
        if n+batch_size < len(data):
            data_batches.append(data[n:n+batch_size])
            label_batches.append(labels[n:n+batch_size])

    if len(data)%batch_size > 0:
        data_batches.append(data[len(data)-(len(data)%batch_size):len(data)])
        label_batches.append(labels[len(data)-(len(data)%batch_size):len(data)])
        
    return data_batches,label_batches


def features2image(feature_vector,G_rev):
    '''reconstruct image using feature_vector and Gabor wavelet filters'''

    # shape of G_rev = (nPixels,nFeatures)
    # recon_im_vector.shape = (nPixels,nFrames)
    
    recon_im_vector = G_rev @ feature_vector
    reconstructed_im = recon_im_vector.reshape((32,32))

    return reconstructed_im


### plotting fxns
def plot_correlation_hist(rs,color,yticks):
    plt.figure(figsize=np.r_[1.5,1]*4)
    plt.hist(rs,bins=20,range=[0,1],edgecolor="black",linewidth=2,color=color)

    ax = plt.gca()
    ax.tick_params('both',length=6,width=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.xticks([0,0.2,0.4,0.6,0.8,1],fontsize=15)
    plt.xlabel('Pearson\'s Correlation (r)',fontsize=20)
    plt.xlim([-0.05,1.05])

    plt.yticks(yticks,fontsize=15)
    plt.ylabel('Number of Samples',fontsize=20)
    plt.ylim([yticks[0],yticks[-1]])
    
    plt.show()

    
def plot_epoch_mse_folds(fold_mses,test_mses=None):
    plt.figure(figsize=np.r_[2,1]*4)
    
    # plot epoch MSE values
    avg_epoch_mses = np.mean(np.asarray(fold_mses),axis=0)
    if len(fold_mses) > 1:
        for i,epoch_mses in enumerate(fold_mses):
            if i == 1:
                plt.plot(epoch_mses,color='lightgrey',lw=3,label='MSE of Single Fold')
            else:
                plt.plot(epoch_mses,color='lightgrey',lw=3)
        plt.plot(avg_epoch_mses,color='black',lw=3,label='Average MSE')
    elif len(fold_mses) == 1:
        plt.plot(avg_epoch_mses,color='black',lw=3,label='Average MSE of Epochs')
        
    # plot test data MSE values
    avg_test_mse = np.mean(test_mses)
    if len(test_mses) > 1:
        for i,test_mse in enumerate(test_mses):
            x = len(fold_mses[0])
            if i == 1:
                plt.plot(x,test_mse,color='grey',marker='o',markersize=10,label='Avg MSE of Test Data')
            else:
                plt.plot(x,test_mse,color='grey',marker='o',markersize=10)
        xs = [len(fold_mses[0])-0.5,len(fold_mses[0])+0.5]
        plt.plot(xs,[avg_test_mse,avg_test_mse],color='red',lw=3,label='Average MSE of Test Data Across Folds')
    elif len(fold_mses) == 1:
        xs = [len(fold_mses[0])-0.5,len(fold_mses[0])+0.5]
        plt.plot(xs,[test_mses,test_mses],color='red',lw=3,label='Average MSE of Test Data')

    ax = plt.gca()
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax.tick_params('both',length=6,width=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.xticks(list(range(0,avg_epoch_mses.shape[0]+1,2)),labels=list(range(1,avg_epoch_mses.shape[0]+1,2))+['Test'],fontsize=15)
    plt.xlabel('Epoch #',fontsize=20)
    plt.xlim([-0.05,avg_epoch_mses.shape[0]+1])

    plt.yticks(fontsize=15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    plt.ylabel('MSE',fontsize=20)

    plt.legend(fontsize=15,loc='upper right',edgecolor='white')
    
    plt.show()
    
    
def plot_frames_pred(im,feat,G_rev,pred_im,iF,rep=0):
    idx = im.shape[2]*rep + iF

    recon_im = features2image(feat[idx,:],G_rev)

    fig = plt.figure(figsize=np.r_[3,1]*5)
    
    fig.add_subplot(1,3,1)
    plt.imshow(im[:,:,iF])
    plt.title('Original Im.')
    plt.axis('off')
    
    fig.add_subplot(1,3,2)
    plt.imshow(recon_im)
    plt.title('Recon. Im. from Real FV')
    plt.axis('off')
    
    fig.add_subplot(1,3,3)
    plt.imshow(pred_im[idx,:,:])
    plt.title('Recon. Im. from Predicted FV')
    plt.axis('off')
    
    plt.show()
    
    
### artificial perturbation example code
def artificial_perturb_predict(im,G_rev,resp,model,perturb_IDs,perturb_frames,perturb_value,save_path):
    perturb_resp = resp.copy()
    perturb_resp[perturb_frames[0]:perturb_frames[1],perturb_IDs] = perturb_value
        
    # generate move with perturbed response
    perturb_im = np.zeros((resp.shape[0],im.shape[0],im.shape[1]))
    for iF in range(resp.shape[0]):
        # calculate predicted feature vector from model
        pred_fv = np.asarray(model(torch.tensor(perturb_resp[iF,:].astype(np.float32))).data)
        # calculate the predicted image
        pred_im = features2image(pred_fv,G_rev)

        perturb_im[iF,:,:] = pred_im

    save_name = os.path.join(save_path,'perturbed_movie.tif')
    tfl.imsave(save_name,perturb_im,bigtiff=True)
    
    return perturb_resp,perturb_im


def plot_artificial_perturb_trace(perturb_resp,resp,cell_ID):
    iC = cell_ID

    plt.figure(figsize=np.r_[1.5,1]*5)

    plt.plot(perturb_resp[:600,iC],color='red',lw=2.5,label='Perturbation')
    plt.plot(resp[:600,iC],color='black',lw=3,label='Response of Cell')

    ax = plt.gca()
    ax.tick_params('both',length=6,width=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.xticks([0,100,200,300,400,500,600],fontsize=15)
    plt.xlabel('Frame in Movie',fontsize=20)
    plt.xlim([0,600])

    plt.yticks([-0.3,0,0.3],fontsize=15)
    plt.ylabel('Activation',fontsize=20)
    plt.ylim([-0.3,0.3])

    plt.legend(fontsize=15,edgecolor='white')

    plt.show()