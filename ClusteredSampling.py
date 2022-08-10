import numpy as np
import pandas as pd

# df = pd.DataFrame({'id': ['11', '12', '13', '14', '15', '21', '22', '23', '24', '25', '31', '32', '33', '34', '35'],
#                    'event_type': ['type1', 'type1', 'type1', 'type1', 'type1', 'type2', 'type2', 'type2', 'type2',
#                                   'type2', 'type3', 'type3', 'type3', 'type3',
#                                   'type3']})

N = 10000
# Generating Population data
price_vb = pd.Series(np.random.uniform(1, 4, size=N))
id = pd.Series(np.arange(0, len(price_vb), 1))
event_type = pd.Series(np.random.choice(["type1", "type2", "type3"], size=len(price_vb)))
click = pd.Series(np.random.choice([0, 1], size=len(price_vb)))
df = pd.concat([id, price_vb, event_type, click], axis=1)
df.columns = ["id", "price", "event_type", "click"]
df


def get_clustered_Sample(df, n_per_cluster, num_select_clusters, cluster_by):
    N = len(df)
    K = int(N / n_per_cluster)
    data = None

    keys, values = df.sort_values('event_type')[['event_type', 'id']].values.T
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index[1:])
    clustered_df = pd.DataFrame({'Case_Status': ukeys, 'ID': [list(a) for a in arrays]})

    for case, case_id in clustered_df.iterrows():
        print(case, case_id)

    for case, case_ids in clustered_df:
        sample_k = df[df.event_type == k]
        sample_k = sample_k.sample(n_per_cluster)
        sample_k["cluster"] = np.repeat(k, len(sample_k))
        df = df.drop(index=sample_k.index)
        data = pd.concat([data, sample_k], axis=0)

    random_chosen_clusters = np.random.randint(0, K, size=num_select_clusters)
    samples = data[data.cluster.isin(random_chosen_clusters)]
    return samples


sample = get_clustered_Sample(df=df, n_per_cluster=100, num_select_clusters=2, cluster_by=event_type.drop_duplicates())
sample
