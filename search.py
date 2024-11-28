import pandas as pd
import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.init import xavier_uniform_
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set_theme()
#not sure which one is necessary
np.random.seed(42)
torch.manual_seed(42)




#load data
df_train = pd.read_csv('./data/train.csv')

book_ids = df_train['book_id'].unique()
user_ids = df_train['user_id'].unique()

n_books = len(book_ids)
n_users = len(user_ids)

#unique index for each user and book in the matrix
book_idx = {ids: i for i,ids in enumerate(book_ids)}
user_idx = {ids: i for i,ids in enumerate(user_ids)}

### whole df to train on once optimal hyperparameters are found

#to access vectors in MF
user_assigned_idx = torch.LongTensor(np.array([user_idx[i] for i in df_train['user_id'].values]))
book_assigned_idx = torch.LongTensor(np.array([book_idx[i] for i in df_train['book_id'].values]))

# to compute training loss
ratings = torch.FloatTensor(df_train['rating'].values)


#train-val split for hyperparam selection
df_train['user_idx'] = df_train['user_id'].map(user_idx)
df_train['book_idx'] = df_train['book_id'].map(book_idx)

#we keep a low validation size to capture most user-book interactions
train_data, val_data = train_test_split(df_train, test_size=0.01, random_state=42)

user_assigned_idx_train = torch.LongTensor(train_data['user_idx'].values)
book_assigned_idx_train = torch.LongTensor(train_data['book_idx'].values)
ratings_train = torch.FloatTensor(train_data['rating'].values)

user_assigned_idx_val = torch.LongTensor(val_data['user_idx'].values)
book_assigned_idx_val = torch.LongTensor(val_data['book_idx'].values)
ratings_val = torch.FloatTensor(val_data['rating'].values)



class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_books, embedding_size, var, init):
        """
        Initializes a MatrixFactorization model
        P contains the embedded vectors of users
        Q contains the embedded vectors of books

        input:
            n_users: int, number of users
            n_books: int, number of books
            embedding_size: int, dimension of embedded space
            var: float, range of initialized weights (for random or uniform initialization)
            init: string, type of method to initialize weights, must be in {uniform, normal xavier}, keeps initialization from nn.Embeding otherwise
        """


        super().__init__()
        self.P = nn.Embedding(n_users, embedding_size)
        self.Q = nn.Embedding(n_books, embedding_size)

        #change weights initialization:

        if init == 'uniform':
            self.P.weight.data.uniform_(0, var)
            self.Q.weight.data.uniform_(0, var)

        if init == 'normal':
            self.P.weight.data.normal_(mean=0, std=var)
            self.Q.weight.data.normal_(mean=0, std=var)

        if init == 'xavier':
            xavier_uniform_(self.P.weight)
            xavier_uniform_(self.Q.weight)


        
    def forward(self, user_id, book_id):
        """
        Forward pass to predict ratings

        inputs:
            user_id: tensor, ids of user
            book_id: tensor, ids of books
        ouput:
            out: tensor, predicted ratings of (user, book) pairs
        """

        user_vec = self.P(user_id)
        book_vec = self.Q(book_id)
        #dot product
        out = (user_vec*book_vec).sum(1)
        return out


    

#metric
mse_metric = torch.nn.MSELoss()

def train(user_assigned_idx, book_assigned_idx, ratings, embedding_size, var, init, decay, lr, lambda_, N_EPOCH, verbose):
    """
    Trains a MF model given the hyperparameters

    Input:
        user_assigned_idx: array containing the unique user id to train on
        book_assigned_idx: array containing the unique book id to train on
        ratings: array containing ratings to train on
        embedding_size: int, dimension of embedded space
        var: float, range of values for weight initialization
        init: string: type of weight initialization
        decay: float: l2 regularization implemented in Adam optimizer
        lr: float: optimizer's learning rate
        lambda_: float (not used): regularization to aim for 2.5 mean rating
        N_EPOCH: int, number of epoch to train on
        verbose: boolean: whether to print rmse and mean rating during training

    Output:
        model: a trained MF model
    """
    
    model = MatrixFactorization(n_users, n_books, embedding_size, var, init)

    #weight decay acts as a l2 regularizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    model.train()

    for epoch in range(N_EPOCH):

        optimizer.zero_grad()

        #prediction
        r_hat = model(user_assigned_idx, book_assigned_idx)

        mse = mse_metric(r_hat, ratings)
        rmse = torch.sqrt(mse)
        #adds regularization term to aim for mean rating
        loss = mse + lambda_*(torch.mean(r_hat) - 2.5)**2
        
        #update
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0 and verbose:
            print(f'epoch {epoch+1}, RMSE: {rmse.item()}, mean rating: {torch.mean(r_hat)}')

    return model


def validation(model, user_assigned_idx_val, book_assigned_idx_val, ratings):
    """
    Estimates error of a trained model on unseen data
    
    Input:
        model: MF object, a trained model
        user_assigned_idx_val: array of unique user id of the validation set
        book_assigned_idx_val: array of unique book id of the validation set
        ratings: array of ratings of the validation set

    Output:
        rmse: float, rmse on validation set
        mean_rating: float, mean predicted rating
    """


    #evaluation mode as training is done
    model.eval()
    #predictions
    r_hat = model(user_assigned_idx_val, book_assigned_idx_val)
    #get ratings between 0 and 5 to lower possible error
    r_hat_clipped = torch.clamp(r_hat, 0, 5)

    err = mse_metric(r_hat_clipped, ratings)
    rmse = torch.sqrt(err)
    mean_rating = torch.mean(r_hat_clipped)

    return rmse.item(), mean_rating




###Grid search for d and l_2 reg###

ds = [16, 32, 64, 128, 256, 512]
vars = 1e-4
decays = [0, 1e-7, 1e-6, 1e-5]

losses = np.loadtxt('./saved_arrays/losses_final_inshallah.txt')
means = np.loadtxt('./saved_arrays/means_final_inshallah.txt')

map_losses = losses.reshape((len(ds), len(decays)))
map_means = means.reshape((len(ds), len(decays)))

fig, axes = plt.subplots(2, 1, figsize=(40, 40))

sns.heatmap(map_losses, annot=True, fmt=".2f", xticklabels=decays, yticklabels=ds, cmap="viridis", ax=axes[0], annot_kws={"size": 25})
axes[0].set_title("Heatmap of RMSE for different hyperparameters", fontsize=30)
axes[0].set_xlabel(r'$\ell_2$ regularizer', fontsize=25)
axes[0].set_ylabel(r'$d$', fontsize=25)
axes[0].tick_params(axis='x', labelsize=25) 
axes[0].tick_params(axis='y', labelsize=25)  

sns.heatmap(map_means, annot=True, fmt=".2f", xticklabels=decays, yticklabels=ds, cmap="viridis", ax=axes[1], annot_kws={"size": 25})
axes[1].set_title("Heatmap of mean rating for different hyperparameters", fontsize=30)
axes[1].set_xlabel(r'$\ell_2$ regularizer', fontsize=25)
axes[1].set_ylabel(r'$d$', fontsize = 25)
axes[1].tick_params(axis='x', labelsize=25) 
axes[1].tick_params(axis='y', labelsize=25) 

plt.savefig('./plot/grid_torch.pdf')
plt.show()





###Weights initialization comparison ###

vars = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
N_epoch = 2000

def plot_init(init, vars):
    gamma = 1e-4
    reg = 1e-6
    d = 50
    

    losses = np.zeros(len(vars))
    means = np.zeros(len(vars))

    for i,var in enumerate(vars):
        print(var)
        model = train(user_assigned_idx_train, book_assigned_idx_train, ratings_train, d, var, init, reg, gamma, 0, N_epoch, False)
        losses[i], means[i] = validation(model, user_assigned_idx_val, book_assigned_idx_val, ratings_val)


    return losses, means

        

losses_uni, means_uni = plot_init('uniform', vars)
losses_norm, means_norm = plot_init('normal', vars)
losses_xav, means_xav = plot_init('xavier', vars)




fig, axes = plt.subplots(2, 1, figsize=(15, 13))
axes[0].plot(vars, losses_uni, label=r'$\mathcal{U}$')
axes[0].plot(vars, losses_xav, label='Xavier')
axes[0].plot(vars, losses_norm, label=r'$\mathcal{N}$')
axes[0].set_title("RMSE vs variance")
axes[0].set_xlabel(r'$c$')
axes[0].set_ylabel('RMSE')
axes[0].legend(loc='upper right')
axes[0].set_xscale('log')

axes[1].plot(vars, means_uni, label=r'$\mathcal{U}$')
axes[1].plot(vars, means_xav, label='Xavier')
axes[1].plot(vars, means_norm, label=r'$\mathcal{N}$')
axes[1].set_title(f'Mean rating vs variance ({N_epoch} epochs)')
axes[1].set_xlabel(r'$c$')
axes[1].set_ylabel('Mean rating')
axes[1].legend(loc='upper right')
axes[1].set_xscale('log')

plt.savefig('./plot/weight_init.pdf')
plt.show()




### plot to find d_star ### 

N_EPOCH = 2000
lr = 1e-4
lambda_ = 0
init = 'uniform'
verbose = False
ds = list(range(50, 260, 10))
var = 0.1
dec = 1e-6


losses = np.zeros(len(ds))
means = np.zeros(len(ds))
for i,d in enumerate(ds):
    print(d)
    nepoch = 1000 if d >= 120 else 2000
    model = train(user_assigned_idx_train, book_assigned_idx_train, ratings_train, d, var, init, dec, lr, lambda_, nepoch, verbose)
    losses[i], means[i] = validation(model, user_assigned_idx_val, book_assigned_idx_val, ratings_val)



fig, axes = plt.subplots(2, 1, figsize=(15, 15))
ds = list(range(50, 260, 10))
axes[0].plot(ds, losses)
axes[0].set_title("RMSE vs embedding dimension")
axes[0].set_xlabel(r'$d$')
axes[0].set_ylabel('RMSE')


axes[1].plot(ds, means)
axes[1].set_title("Mean rating vs embedding dimension")
axes[1].set_xlabel(r'$d$')
axes[1].set_ylabel('Mean rating')

plt.savefig('./plot/d_star.pdf')
plt.show()





