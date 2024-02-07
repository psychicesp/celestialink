from datetime import datetime as dt

import os
import pandas as pd
import phenograph
from sklearn.preprocessing import StandardScaler

from src.segmented_svc.segmented_svc import SegmentedSVC

test_df = pd.read_csv(os.path.join("celestialink","test", "flow_cytometry_test_data.csv"))

run_times = []

w = 0

x = 200000

scaler = StandardScaler()
training_df = test_df.sample(250000)
scaled_df = scaler.fit_transform(training_df)

communities, graph, Q  = phenograph.cluster(
    scaled_df,
    clustering_algo = "leiden",
    )

celestia_object = SegmentedSVC(
    data = training_df.values,
    labels = communities
    )

t1 = dt.now()

while x < len(test_df)+200000:
    sub_df = test_df[w:x]
    run_dict = {}
    run_dict['database size'] = x
    
    
    _ = celestia_object.predict(sub_df.values)
    t2 = dt.now()
    time_elapsed = t2-t1
    print(f"{x} rows completed in {time_elapsed.seconds} seconds")
    run_dict['SegmentedSVC'] = time_elapsed.seconds
    
    
    run_times.append(run_dict)
    run_df = pd.DataFrame(run_times)
    run_df.to_csv("long run.csv")
    w = x
    x += 200000
