
import os
import pandas as pd
import phenograph
from sklearn.preprocessing import StandardScaler

from src.segmented_svc.segmented_svc import SegmentedSVC

test_df = pd.read_csv(os.path.join("celestialink","test", "flow_cytometry_test_data.csv")).fillna(0)

from pprint import pprint
from datetime import datetime as dt

run_times = []

x = 50000

while x < 1000000:
    sub_df = test_df[:x]
    run_dict = {}
    run_dict['database size'] = x
    scaler = StandardScaler()
    scaled_sub_df = scaler.fit_transform(sub_df)
    
    try:
        t1 = dt.now()
        communities, graph, Q  = phenograph.cluster(
        scaled_sub_df,
        clustering_algo = "leiden",
        )
        t2 = dt.now()
        time_elapsed = t2-t1
        run_dict['Leiden'] = time_elapsed.seconds
    except:
        run_dict['Leiden'] = 0
    
    
    t1 = dt.now()
    celestia_object = SegmentedSVC(
    data = sub_df.values,
    labels = communities
    )
    t2 = dt.now()
    time_elapsed = t2-t1
    run_dict['SegmentedSVC Training'] = time_elapsed.seconds
    
    
    t1 = dt.now()
    _ = celestia_object.predict(sub_df.values)
    t2 = dt.now()
    time_elapsed = t2-t1
    run_dict['SegmentedSVC Predicting'] = time_elapsed.seconds
    
    
    run_times.append(run_dict)
    run_df = pd.DataFrame(run_times)
    run_df.to_csv("big run.csv")
    x += 50000
    if x > 1000000:
        break
