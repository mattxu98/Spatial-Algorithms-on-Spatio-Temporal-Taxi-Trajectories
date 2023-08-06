### Spatial Algorithms on Spatio Temporal Taxi Trajectories
- Improved query efficiency over 10^5 preprocessed trajectories via algorithms. Designed a spatio-temporal database
- Realized R-tree/balltree/kd tree/Hausdorff/DTW distance in Python & SQL, by transforming trajectories and data types
- Compared time/memory costs and performance: precision/recall/f1, among algorithms and linear scan
- Visualization. Algorithms cost less than linear scan (<1/5), both performance >0.85 with few gaps. Optimizing benchmark/distance conversion improved performance. Gained develop efficiency by building belt queries on trajectory knn