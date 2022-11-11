import os
import pandas as pd

os.system("wget https://files.grouplens.org/datasets/movielens/ml-20m.zip")

os.system("unzip ml-20m.zip")

m = pd.read_csv('ml-20m/movies.csv')
m = m[m.movieId.notnull()].reindex()
m['itemid']=m.movieId.apply(lambda x: str(int(x)))
m['product_name'] = m['title']
items = m[['itemid','product_name','genres']]
items.to_json('items.json')

#Purchases and grouped purchases.
interactions = pd.read_csv('ml-20m/ratings.csv')
interactions = interactions[interactions.rating>=4.]
interactions = interactions.sort_values(['userId','timestamp'])
interactions['itemid'] = interactions['movieId'].apply(str)
interactions['userid'] = interactions['userId'].apply(str)
interactions['amount'] = 1
interactions['date'] = interactions['timestamp']
interactions[['itemid','userid','amount','date']]
interactions.to_json("purchases.json")
interactions['itemids'] = interactions[['userid','itemid']].groupby(['userid'])['itemid'].transform(lambda x: ','.join(x))
iii = interactions[['userId','itemids']].drop_duplicates()
iii.to_json('purchases_txt.json')

#Keep only users with 5 or more interactions.
#Purchases.
purchases=pd.read_json('purchases.json')
purchases['userid'] = purchases.userid.apply(str)
purchases['itemid'] = purchases.itemid.apply(str)
purchases_item_counts = purchases[['userid','itemid']]
purchases_user_counts = purchases[['userid','itemid']]
purchases_user_count = purchases.groupby(['userid']).size().to_frame('nr_of_purchases').reset_index()
purchases_user_count = purchases_user_count.sort_values(by=['nr_of_purchases'], ascending=False)
pu5=purchases_user_count[purchases_user_count.nr_of_purchases>=5]
purchases_pu5 = purchases[purchases.userid.isin(pu5.userid)]
purchases_item_count_pu5 = purchases_pu5.groupby(['itemid']).size().to_frame('nr_of_purchases').reset_index()
purchases_item_count_pu5 = purchases_item_count_pu5.sort_values(by=['nr_of_purchases'], ascending=False)
purchases_pu5.to_json('purchases_pu5.json')

#Grouped purchases.
purchases_pu5['itemids'] = purchases_pu5[['userid','itemid']].groupby(['userid'])['itemid'].transform(lambda x: ','.join(x))
iii = purchases_pu5[['userId','itemids']].drop_duplicates()
iii['userid']=iii['userId'].apply(str)
iii = iii[['userid','itemids']]
iii.to_json('purchases_txt_pu5.json')

#Users.
iii['userid'].to_frame().to_json('users_pu5.json')

#Items
items[items.itemid.isin(purchases_item_count_pu5.itemid)].to_json("items_pu5.json")

#Items sorted by number of interactions
purchases_item_count_pu5.to_json("items_sorted_pu5.json")

#Users sorted by number of interactions
pu5.to_json("users_sorted_pu5.json")

#Create train, val and test split.
users           = pd.read_json('users_pu5.json')
shuffled_users  = users.sample(frac=1., random_state=42)
test_users      = shuffled_users.iloc[:10000]
val_users       = shuffled_users.iloc[10000:20000]
train_users     = shuffled_users.iloc[20000:]

test_users.to_json("test_users.json")
val_users.to_json("val_users.json")
train_users.to_json("train_users.json")
