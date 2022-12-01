import sys

import pandas as pd
import numpy as np
import scrapbook as sb
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.rbm.rbm import RBM
from recommenders.datasets.python_splitters import numpy_stratified_split
from recommenders.datasets.sparse import AffinityMatrix
from recommenders.datasets import movielens
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k, rmse, mae
from recommenders.utils.timer import Timer
from recommenders.utils.plot import line_graph

MOVIELENS_DATA_SIZE = '1m'

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=['userID','movieID','rating','timestamp']
)

#to use standard names across the analysis
header = {
        "col_user": "userID",
        "col_item": "movieID",
        "col_rating": "rating",
    }


am = AffinityMatrix(df = data, **header)
X, _, _ = am.gen_affinity_matrix()

Xtr, Xtst = numpy_stratified_split(X)


model = RBM(
    possible_ratings=np.setdiff1d(np.unique(Xtr), np.array([0])),
    visible_units=Xtr.shape[1],
    hidden_units=1200,
    training_epoch=1250,
    minibatch_size=350,
    keep_prob=0.9,
    with_metrics=True
)

with Timer() as train_time:
    model.fit(Xtr)

print("Took {:.2f} seconds for training.".format(train_time.interval))

# number of top score elements to be recommended
K = 10

# Model prediction on the test set Xtst.
with Timer() as prediction_time:
    top_k =  model.recommend_k_items(Xtst)

print("Took {:.2f} seconds for prediction.".format(prediction_time.interval))
top_k_df = am.map_back_sparse(top_k, kind = 'prediction')
test_df = am.map_back_sparse(Xtst, kind = 'ratings')


def ranking_metrics(
        data_size,
        data_true,
        data_pred,
        K
):
    eval_map = map_at_k(data_true, data_pred, col_user="userID", col_item="movieID",
                        col_rating="rating", col_prediction="prediction",
                        relevancy_method="top_k", k=K)

    eval_ndcg = ndcg_at_k(data_true, data_pred, col_user="userID", col_item="movieID",
                          col_rating="rating", col_prediction="prediction",
                          relevancy_method="top_k", k=K)

    eval_precision = precision_at_k(data_true, data_pred, col_user="userID", col_item="movieID",
                                    col_rating="rating", col_prediction="prediction",
                                    relevancy_method="top_k", k=K)

    eval_recall = recall_at_k(data_true, data_pred, col_user="userID", col_item="movieID",
                              col_rating="rating", col_prediction="prediction",
                              relevancy_method="top_k", k=K)
    eval_rmse = rmse(data_true, data_pred, col_user="userID", col_item="movieID", col_rating="rating",
                     col_prediction="prediction")

    df_result = pd.DataFrame(
        {"Dataset": data_size,
         "K": K,
         "MAP": eval_map,
         "nDCG@k": eval_ndcg,
         "Precision@k": eval_precision,
         "Recall@k": eval_recall,
         "RMSE": eval_rmse
         },
        index=[0]
    )

    return df_result

eval_1m = ranking_metrics(
    data_size="mv 1m",
    data_true=test_df,
    data_pred=top_k_df,
    K=10
)

print(eval_1m)