'''
Project: Spatial Algorithms on Spatio Temporal Taxi Trajectories
Author: Longpeng Xu (https://github.com/mattxu98/)
Date: 20230526
'''
# =================================================================
# The algorithms used:
# =================================================================
# 	Algorithm 0 Linear Scan        Task A, B, C, D, E
#	Algorithm 1 R-tree             Task A
#	Algorithm 2 Ball tree          Task B, C
#	Algorithm 3 kd tree            Task D
#	Algorithm 4 Hausdorff distance Task E
#	Algorithm 5 DTW distance       Task E



# =================================================================
# Data Preprocessing
# =================================================================

# -----------------------------------------------------------------
# Calculation of time cost, memory cost, precision, recall, f1
# -----------------------------------------------------------------

import time
import psutil

# Time and memory cost before query - NO need to paste them after every case 
#time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

# Time and memory cost incurred by the query - NO need to paste them after every case 
def after_query():
    time_cost = time.time() - time_before
    memory_cost = psutil.Process().memory_info().rss / 1024**2 - memory_before
    print(f"Time cost: {round(time_cost, 8)} seconds")
    print(f"Memory cost: {round(memory_cost, 8)} megabytes")
    return None

# Precision, recall, f1 - paste them after every case 
#pred_set = # paste the printed result from each case in Python, formatted as list, in some cases remain pointid drop distance
#truth_set = # paste the result from each case in PostgreSQL, formatted as list, in some cases remain pointid drop distance
#true_positive = len([x for x in pred_set if x in truth_set])
#false_positive = len([x for x in pred_set if x not in truth_set])
#false_negative = len([x for x in truth_set if x not in pred_set])
#precision = true_positive / (true_positive + false_positive)
#recall = true_positive / (true_positive + false_negative)
#f1 = 2 * precision * recall / (precision + recall)
#print('precision ', precision, ' recall ', recall, ' f1 ', f1)

# -----------------------------------------------------------------
# Import the original data
# -----------------------------------------------------------------

import pandas as pd
import numpy as np
import datetime

# Preprocessing: downsize the dataset to 100,000 rows (six digits)
taxi = pd.read_csv('train.csv', header=0, nrows=100000)
# Preprocessing: ensure each TRIP_ID has six digits, is unique and is headed by a non-zero digit,
#                so that TRIP_ID ranges from 100,000 to 199,999
taxi['TRIP_ID'] = taxi.index + 100000
# Preprocessing: convert each unix timestamp to readable datetime
for i in range(len(taxi)):
    taxi.loc[i,'TIMESTAMP'] = datetime.datetime.fromtimestamp(taxi.loc[i,'TIMESTAMP'])
taxi

# -----------------------------------------------------------------
# Preprocess the original data for Python use
# -----------------------------------------------------------------

def preprocess(data):
    '''
    Preprocessing by extracting every point (longitude-latitude pair) and assign each
    with a unique id. Assume one 'POLYLINE' corresponds to one 'TRIP_ID' in the data.
    '''
    id_lon_lat = pd.DataFrame(columns = ['tripid','pointid', 'timestamp1', 'stand', 'lon', 'lat'])
    for i in range(len(data)):
        # Preprocessing: For each row of data, extract a list of strings, each string 
        #                is a longitude-latitude pair
        if data.loc[i, 'POLYLINE'] == '[]' or len(data.loc[i, 'POLYLINE']) <= 23:
            pass
        else:
            locs_row = data.loc[i,'POLYLINE'].replace('[[','').replace(']]','').split('],[')
            tripid = data.loc[i,'TRIP_ID']
            timestamp1 = data.loc[i,'TIMESTAMP']
            stand = data.loc[i,'ORIGIN_STAND']
            for j in range(len(locs_row)):
                # Preprocessing: Assign an unique index to each longitude-latitude pair,
                #                by extending the row index with the index of the pair
                #                in the list (i.e., append three more digits)
                pointid = int(str(tripid) + f"{j:03}")
                # Preprocessing: Extract the longitude and the latitude from the pair
                lon_lat = [float(k) for k in locs_row[j].split(',')]
                id_lon_lat = pd.concat([id_lon_lat, \
                                        pd.DataFrame([[tripid, pointid, timestamp1, stand, lon_lat[0], lon_lat[1]]],\
                                        columns=id_lon_lat.columns)], ignore_index=True)
    return id_lon_lat.astype({'tripid': int, 'pointid': int})

taxi_excerpt = preprocess(taxi)
taxi_excerpt

# -----------------------------------------------------------------
# Preprocess the original data for PostgreSQL use
# -----------------------------------------------------------------

def preprocess_sql(data):
    '''Preprocess each trajectory in dataframe 'taxi' to the format accepted by pgAdmin'''
    id_lon_lat = pd.DataFrame(columns = ['tripid','trajectory'])
    
    for i in range(len(data)):        
        # exclude the trajectories that include zero and one point, as
        # pgAdmin can report ERROR:  geometry requires more points
        if data.loc[i, 'POLYLINE'] == "[]" or len(data.loc[i, 'POLYLINE']) <= 23:
            pass
        else:
            # initialize the string of trajectory to the format accepted by pgAdmin
            trajectory = "LINESTRING("
            locs_row = data.loc[i,'POLYLINE'].replace("[[","").replace("]]","").split("],[")
            tripid = data.loc[i,'TRIP_ID']
        
            for j in range(len(locs_row)):
                # reformat longitude, latitude to the format accepted by pgAdmin 
                lon_lat = [k for k in locs_row[j].split(",")]
                lon_lat_str = lon_lat[0] + " " + lon_lat[1] + ", "
                # add the point to the string of trajectory
                trajectory += lon_lat_str
            
            trajectory = trajectory[:-2] + ")"
            
            # add one row of tripid and trajectory each time to build dataset taxi_sql
            id_lon_lat = pd.concat([id_lon_lat, \
                                    pd.DataFrame([[tripid, trajectory]],\
                                    columns=id_lon_lat.columns)], ignore_index=True)
        
    return id_lon_lat.astype({'tripid': int})

taxi_sql = preprocess_sql(taxi)
taxi_sql



# =================================================================
# Task A Retrieve the points in a specified rectangular area and within a given time window
# =================================================================

# -----------------------------------------------------------------
# Algorithm 0 Linear scan
# -----------------------------------------------------------------

def linear_area_time(left, bottom, right, top, start, end, data=taxi_excerpt):
    '''Linear scan to return the pointid of data points in a given rectangular
       area and within a certain time window.
    '''
    assert left < right and bottom < top,\
           "'left'/'bottom' must be smaller than 'right'/'top'"
    assert type(start) == datetime.datetime and type(end) == datetime.datetime,\
           "'start' and 'end' must be datetime.datetime object"
    
    res = []
    
    for i in data.index:
        if  left < data.loc[i,'lon'] and data.loc[i,'lon'] < right\
        and bottom < data.loc[i,'lat'] and data.loc[i,'lat'] < top\
        and (data[data.pointid == data.loc[i,'pointid']].timestamp1 > start).bool()\
        and (data[data.pointid == data.loc[i,'pointid']].timestamp1 < end).bool():
            res += [data.loc[i, 'pointid']]
    return res

# -----------------------------------------------------------------
# Cases
# -----------------------------------------------------------------

# Task A Case 1 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

temp_res = linear_area_time(-8.59, 41.13, -8.54, 41.17,\
                        datetime.datetime(2013,7,1,10,0,0), datetime.datetime(2013,7,1,10,5,0))
print(temp_res)

# Time and memory cost incurred by the query    
after_query()

# Task A Case 2 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

temp_res = linear_area_time(-8.59, 41.13, -8.54, 41.17,\
                        datetime.datetime(2013,7,1,11,10,0), datetime.datetime(2013,7,1,11,15,0))
print(temp_res)

# Time and memory cost incurred by the query    
after_query()

# Task A Case 3 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

temp_res = linear_area_time(-8.65, 41.15, -8.58, 41.20,\
                        datetime.datetime(2013,7,1,10,0,0), datetime.datetime(2013,7,1,10,5,0))
print(temp_res)

# Time and memory cost incurred by the query    
after_query()

# Task A Case 4 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

temp_res = linear_area_time(-8.65, 41.15, -8.58, 41.20,\
                        datetime.datetime(2013,7,1,11,10,0), datetime.datetime(2013,7,1,11,15,0))
print(temp_res)

# Time and memory cost incurred by the query    
after_query()

# -----------------------------------------------------------------
# Algorithm 1 R-tree
# -----------------------------------------------------------------

from rtree import index

def rtree_area_time(left, bottom, right, top, start, end, data=taxi_excerpt):
    '''Build an r-tree by inserting locations identified by pointid.
       Return the pointid in a given rectangular area and within a
       certain time window.
    '''
    assert left < right and bottom < top,\
           "'left'/'bottom' must be smaller than 'right'/'top'"
    assert type(start) == datetime.datetime and type(end) == datetime.datetime,\
           "'start' and 'end' must be datetime.datetime object"
    
    # build an r-tree
    idx = index.Index()
    
    # insert points into the r-tree
    for i in data.index:
        pointid, lon, lat = taxi_excerpt.loc[i, ['pointid','lon','lat']]
        idx.insert(id=pointid, coordinates=(lon, lat, lon, lat))
    
    # check the bounds of the r-tree
    # print(idx)
    
    in_area = list(idx.intersection((left, bottom, right, top)))
    
    return sorted([point for point in in_area if \
                   (data[data.pointid == point].timestamp1 > start).bool() and \
                   (data[data.pointid == point].timestamp1 < end).bool()])
				   
# -----------------------------------------------------------------
# Cases
# -----------------------------------------------------------------

# Task A Case 1 R-tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

temp_res = rtree_area_time(-8.59, 41.13, -8.54, 41.17,\
                        datetime.datetime(2013,7,1,10,0,0), datetime.datetime(2013,7,1,10,5,0))
print(temp_res)

# Time and memory cost incurred by the query    
after_query()

# Task A Case 2 R-tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

temp_res = rtree_area_time(-8.59, 41.13, -8.54, 41.17,\
                        datetime.datetime(2013,7,1,11,10,0), datetime.datetime(2013,7,1,11,15,0))
print(temp_res)

# Time and memory cost incurred by the query    
after_query()

# Task A Case 3 R-tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

temp_res = rtree_area_time(-8.65, 41.15, -8.58, 41.20,\
                        datetime.datetime(2013,7,1,10,0,0), datetime.datetime(2013,7,1,10,5,0))
print(temp_res)

# Time and memory cost incurred by the query    
after_query()

# Task A Case 4 R-tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

temp_res = rtree_area_time(-8.65, 41.15, -8.58, 41.20,\
                        datetime.datetime(2013,7,1,11,10,0), datetime.datetime(2013,7,1,11,15,0))
print(temp_res)

# Time and memory cost incurred by the query    
after_query()



# =================================================================
# Task B Access k nearest neighbours (i.e., points) of a specified trajectory
# =================================================================

# -----------------------------------------------------------------
# Algorithm 0 Linear scan
# -----------------------------------------------------------------

import numpy as np
from haversine import haversine

def linear_knn_tripid(k, tripid):
    '''
    Find k nearest neighbour(s) (i.e., points) of a given trajectory (tripid),
    based on linear scan.
    '''
    
    # retrieve all the points of the trajectory
    tripid_rows = taxi_excerpt[taxi_excerpt['tripid'] == tripid].index
    tripid_points_array = [(taxi_excerpt.loc[i, 'lon'], taxi_excerpt.loc[i, 'lat'])\
                          for i in tripid_rows]
    
    # get the other points by truncating the points of the trajectory
    other_points = taxi_excerpt.loc[np.setdiff1d(taxi_excerpt.index, tripid_rows),\
                                    ['pointid','lon','lat']].reset_index(drop=True)
    other_points_array = [(other_points.loc[i, 'lon'], other_points.loc[i, 'lat'])\
                          for i in other_points.index]
    
    # find the min distance from each of the other points to the trajectory
    dist_i = []
    for i in other_points.index:
        dist_j = []
        for j in range(len(tripid_points_array)):
            dist_j += [haversine(other_points_array[i], tripid_points_array[j], unit='m')]
        dist_i += [min(dist_j)]
    
    # Find the k nearest points to the trajectory
    ind_dist = dict(zip(other_points.index, dist_i))
    res = dict(sorted(ind_dist.items(), key = lambda x: x[1])[:k])
    
    # replace the default indices of the k nearest neighbours in other_points by pointid
    knn_res_pointid = [other_points.loc[k,'pointid'] for k in res.keys()]
    return dict(zip(knn_res_pointid, res.values()))
	
# -----------------------------------------------------------------
# Cases
# -----------------------------------------------------------------

# Task B Case 1 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

knn_res = linear_knn_tripid(20,100000)
print(knn_res)

# Time and memory cost incurred by the query    
after_query()

# Task B Case 2 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

knn_res = linear_knn_tripid(40,100000)
print(knn_res)

# Time and memory cost incurred by the query    
after_query()

# Task B Case 3 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

knn_res = linear_knn_tripid(20,100035)
print(knn_res)

# Time and memory cost incurred by the query    
after_query()

# Task B Case 4 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

knn_res = linear_knn_tripid(40,100035)
print(knn_res)

# Time and memory cost incurred by the query    
after_query()

# -----------------------------------------------------------------
# Algorithm 2 Ball tree (knn query)
# -----------------------------------------------------------------

import numpy as np
from sklearn.neighbors import BallTree
from haversine import haversine

def knn_tripid(k, tripid):
    '''
    Find k nearest neighbour(s) (i.e., points) of a given trajectory (tripid),
    based on a ball tree.
    '''
    # tripid_rows = the indices of the points under the tripid
    tripid_rows = taxi_excerpt[taxi_excerpt['tripid'] == tripid].index
    
    # construct ball tree over the points of the trajectory, with great-circle distance
    balltree = BallTree(taxi_excerpt.loc[tripid_rows,['lon','lat']], metric='haversine')
    
    # get the query points by truncating the points of the trajectory
    query_points = taxi_excerpt.loc[np.setdiff1d(taxi_excerpt.index, tripid_rows),\
                                    ['pointid','lon','lat']].reset_index(drop=True)
    query_points_array = [[query_points.loc[i, 'lon'], query_points.loc[i, 'lat']]\
                          for i in query_points.index]
    
    # compute the shortest great-circle distance of each point to the trajectory
    distances, indices = balltree.query(query_points_array, k=1)
    
    # Find the k nearest points to the trajectory
    dist_unsorted = distances.reshape(1,-1)[0]
    ind_dist = dict(zip(query_points.index, dist_unsorted))
    res = dict(sorted(ind_dist.items(), key = lambda x: x[1])[:k])
    
    # replace the default indices of the k nearest neighbours in taxi_excerpt by pointid
    knn_res_pointid = [query_points.loc[k,'pointid'] for k in res.keys()]
    
    # convert the distances in degree returned by ball tree to meters
    knn_res_dist = [i * 6371.009 * 1000 * np.pi / 180 for i in res.values()]
    
    # return pointid and distance for a knn query, where distance can be re-used later
    return dict(zip(knn_res_pointid, knn_res_dist))

# -----------------------------------------------------------------
# Cases
# -----------------------------------------------------------------

# Task B Case 1 Ball tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

knn_res = knn_tripid(20,100000)
print(knn_res)

# Time and memory cost incurred by the query    
after_query()

# Task B Case 2 Ball tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

knn_res = knn_tripid(40,100000)
print(knn_res)

# Time and memory cost incurred by the query    
after_query()

# Task B Case 3 Ball tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

knn_res = knn_tripid(20,100035)
print(knn_res)

# Time and memory cost incurred by the query    
after_query()

# Task B Case 4 Ball tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

knn_res = knn_tripid(40,100035)
print(knn_res)

# Time and memory cost incurred by the query    
after_query()



# =================================================================
# Task C Retrieve the points within a certain distance to a specified trajectory
# =================================================================

# -----------------------------------------------------------------
# Algorithm 0 Linear scan
# -----------------------------------------------------------------

def linear_within_dist(tripid, dist, k=1, delta_k=1):
    '''
    Find all data points within certain distance to a trajectory using the
    function linear_knn_tripid(k, tripid), the nearest neighbor search on 
    linear scan
    --------------------------------------------------------------------------
    Note: 1. Seting k = 1, delta_k = 1 and using the function linear_knn_tripid 
             of linear scan in Task 2, is the sense of linear scan for this task 
          2. What is returned by linear_within_dist is fixed when tripid and dist
             are fixed, regardless of the changing k and/or delta_k
    '''
    knn_res_list = list(linear_knn_tripid(k, tripid).items())

    if knn_res_list[-1][-1] >= dist:
        while knn_res_list[-1][-1] > dist:
            knn_res_list = knn_res_list[:-1]

    elif knn_res_list[-1][-1] < dist:
        knn_res_list = linear_within_dist(tripid, dist, k + delta_k)

    return knn_res_list
	
# -----------------------------------------------------------------
# Cases
# -----------------------------------------------------------------

# Task C Case 1 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_within_dist(tripid=100000, dist=80))

# Time and memory cost incurred by the query    
after_query()

# Task C Case 2 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_within_dist(tripid=100000, dist=110))

# Time and memory cost incurred by the query    
after_query()

# Task C Case 3 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_within_dist(tripid=100024, dist=80))

# Time and memory cost incurred by the query    
after_query()

# Task C Case 4 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_within_dist(tripid=100024, dist=110))

# Time and memory cost incurred by the query    
after_query()

# -----------------------------------------------------------------
# Algorithm 2 Ball tree (Within-distance-to-trajectory query)
# -----------------------------------------------------------------

def within_dist(tripid, dist, k=10, delta_k=5):
    '''
    Find all data points within certain distance to a trajectory using the
    function knn_tripid(k, tripid), the nearest neighbor search on a ball tree
    --------------------------------------------------------------------------
    dist   : the distance from the trajectory, in meters
    k      : the number of nearest neighbors
    delta_k: the increase added to k, before all the points required are found
    --------------------------------------------------------------------------
    Note: What is returned by within_dist is fixed when tripid and dist are
          fixed, regardless of the changing k and/or delta_k
    '''
    knn_res_list = list(knn_tripid(k, tripid).items())

    if knn_res_list[-1][-1] >= dist:
        while knn_res_list[-1][-1] > dist:
            knn_res_list = knn_res_list[:-1]

    elif knn_res_list[-1][-1] < dist:
        knn_res_list = within_dist(tripid, dist, k + delta_k) # recursion

    return knn_res_list
	
# -----------------------------------------------------------------
# Cases
# -----------------------------------------------------------------

# Task C Case 1 Ball tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(within_dist(tripid=100000, dist=80))

# Time and memory cost incurred by the query    
after_query()

# Task C Case 2 Ball tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(within_dist(tripid=100000, dist=110))

# Time and memory cost incurred by the query    
after_query()

# Task C Case 3 Ball tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(within_dist(tripid=100024, dist=80))

# Time and memory cost incurred by the query    
after_query()

# Task C Case 4 Ball tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(within_dist(tripid=100024, dist=110))

# Time and memory cost incurred by the query    
after_query()



# =================================================================
# Task D Access the points in a circle area centred at a given taxi stand
# =================================================================

# find the points of the trajectories started from a taxi stand
taxi_stand = taxi_excerpt[pd.isna(taxi_excerpt['stand']) == False]

# extract the last three digits of each pointid ('000' indicates a starting point)
taxi_stand_ind = dict(zip(taxi_stand.index, [str(taxi_stand.loc[i, 'pointid'])[-3:] for i in taxi_stand.index]))

# extract the row index of the taxi stands in 'taxi_stand' (also in 'taxi_excerpt')
taxi_stand_start_ind = {k: v for k, v in taxi_stand_ind.items() if v == '000'}

# extract the information of the taxi stands
taxi_stand_start = taxi_stand.loc[taxi_stand_start_ind.keys(), ['pointid', 'stand', 'lon', 'lat']]
taxi_stand_start

# -----------------------------------------------------------------
# Algorithm 0 Linear scan
# -----------------------------------------------------------------

from haversine import haversine
import numpy as np
import pandas as pd

def linear_ids_in_circ(pointid, dist):
    '''
    Find all the data points in a circle area centered at a taxi stand, 
    with the radius 'dist', in meters, by linear scan
    '''
    assert pointid in list(taxi_stand_start['pointid']),\
           "'pointid' must be from dataframe 'taxi_stand_start'"
    ind_within_dist = []
    res = []
    
    # extract the longitude and latitude of the queried pointid
    lon, lat = np.array(taxi_excerpt.loc[taxi_excerpt['pointid'] == pointid,['lon','lat']])[0]
    
    # extract the points from taxi_excerpt
    points_array = [(taxi_excerpt.loc[i, 'lon'], taxi_excerpt.loc[i, 'lat'])\
                    for i in taxi_excerpt.index]
    
    # find the default indices of the data points in the circle area centered at pointid
    for i in range(len(taxi_excerpt)):
        if haversine(points_array[i], (lon, lat), unit='m') <= dist:
            ind_within_dist += [i]
    
    # convert the default indices to the pointids
    for j in sorted(ind_within_dist):
        res += [taxi_excerpt.loc[j, 'pointid']]
    
    return res

# -----------------------------------------------------------------
# Cases
# -----------------------------------------------------------------

# Task D Case 1 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_ids_in_circ(100001000, 200))

# Time and memory cost incurred by the query    
after_query()

# Task D Case 2 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_ids_in_circ(100001000, 500))

# Time and memory cost incurred by the query    
after_query()

# Task D Case 3 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_ids_in_circ(100048000, 200))

# Time and memory cost incurred by the query    
after_query()

# Task D Case 4 Linear scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_ids_in_circ(100048000, 500))

# Time and memory cost incurred by the query    
after_query()

# -----------------------------------------------------------------
# Algorithm 3 kd tree
# -----------------------------------------------------------------

from scipy.spatial import KDTree

def ids_in_circ(pointid, dist):
    '''
    Find all the data points in a circle area centered at a taxi stand, 
    with the radius 'dist', in meters, by a kd tree
    '''
    assert pointid in list(taxi_stand_start['pointid']),\
           "'pointid' must be from dataframe 'taxi_stand_start'"
    
    # Construct a kdtree by the coordinates
    kdtree = KDTree(taxi_excerpt[['lon','lat']], balanced_tree=True)
    
	# Obtain the lon-lat pair of the pointid
    point = np.array(taxi_stand_start[taxi_stand_start['pointid'] == pointid][['lon','lat']])[0]
    
	# Query the KDTree for the circle centered at the taxi stands
    circled_points = kdtree.query_ball_point(point, dist/(6371.009*1000) * 180/np.pi)
    
	# Get the IDs of the points within the circle area
    return sorted([taxi_excerpt.loc[i, 'pointid'] for i in circled_points])

# -----------------------------------------------------------------
# Cases
# -----------------------------------------------------------------

# Task D Case 1 kd-tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(ids_in_circ(100001000, 200))

# Time and memory cost incurred by the query    
after_query()

# Task D Case 2 kd-tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(ids_in_circ(100001000, 500))

# Time and memory cost incurred by the query    
after_query()

# Task D Case 3 kd-tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(ids_in_circ(100048000, 200))

# Time and memory cost incurred by the query    
after_query()

# Task D Case 4 kd-tree

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(ids_in_circ(100048000, 500))

# Time and memory cost incurred by the query    
after_query()



# =================================================================
# Task E Retrieve k most similar trajectories of a queried trajectory
# =================================================================

# -----------------------------------------------------------------
# Algorithm 0 Linear scan
# -----------------------------------------------------------------

# Regard the queried trajectory as the outer layer, 
# i.e., find max distance across the queried trajectory points.

def linear_most_sim_tr(tripid, k, data=taxi_excerpt):
    '''1. Find k most similar trajectories of a queried trajectory
          by linear scan manner - starting from scratch and without
          using package scipy.spatial.distance.directed_hausdorff.
	   2. Regard the queried trajectory as the outer layer, 
	      i.e., find max distance across the queried trajectory points.	
       3. The smaller the Hausdorff distance of a trajectory, the
          greater the similarity of it to the queried trajectory. 
    '''
    other_tr = []
    other_tripid = []
    
    # extract the queried trajectory by tripid
    queried_ind = data[data['tripid'] == tripid].index
    queried = [(data.loc[i, 'lon'], data.loc[i, 'lat']) for i in queried_ind]
    
    # extract the other trajectory by the other tripid's    
    for i in set(data['tripid']).difference({tripid}):
        tr_tripid_ind = data[data['tripid'] == i].index
        tr_tripid = [(data.loc[j, 'lon'], data.loc[j, 'lat']) for j in tr_tripid_ind]
        other_tr += [tr_tripid]
        other_tripid += [i]
    
    # calculate the Hausdorff distance by definition 'max(min(i,j))'
    dist_res = []
    for i in range(len(other_tripid)):
        tr_1 = queried
        tr_2 = other_tr[i]
        outer = []
        for point1 in tr_1:
            inner = []
            for point2 in tr_2:
                inner += [np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)]
            outer += [min(inner)]
        dist_res += [max(outer)]
    
    # return the 20 most similar trajectories, in terms of tripid and Hausdorff distance
    tripid_dist = dict(zip(other_tripid, dist_res))
    return dict(sorted(tripid_dist.items(), key = lambda x: x[1])[:k])
	
# -----------------------------------------------------------------
# Cases
# -----------------------------------------------------------------

# Task 5 Case 1 Linear Scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_most_sim_tr(100000, 20))

# Time and memory cost incurred by the query    
after_query()

# Task 5 Case 2 Linear Scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_most_sim_tr(100000, 30))

# Time and memory cost incurred by the query    
after_query()

# Task 5 Case 3 Linear Scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_most_sim_tr(100098, 20))

# Time and memory cost incurred by the query    
after_query()

# Task 5 Case 4 Linear Scan

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(linear_most_sim_tr(100098, 30))

# Time and memory cost incurred by the query    
after_query()

# -----------------------------------------------------------------
# Algorithm 4 Hausdorff distance
# -----------------------------------------------------------------

from scipy.spatial.distance import directed_hausdorff
import numpy as np

# Regard the queried trajectory as the outer layer, 
# i.e., find max distance across the queried trajectory points.

def hsdf_most_sim_tr(tripid, k, data=taxi_excerpt):
    '''Find a trajectory that is most similar to a queried trajectory,
       by the algorithm of Hausdorff distance.The smaller the Hausdorff
       distance of a trajectory, the greater the similarity of it to
       the queried trajectory. 
    '''
    other_tr = []
    other_tripid = []
    dist = []
    
    # extract the queried trajectory by tripid
    queried_ind = data[data['tripid'] == tripid].index
    queried = [(data.loc[i, 'lon'], data.loc[i, 'lat']) for i in queried_ind]
    
    # extract the other trajectory by the other tripid's    
    for i in set(data['tripid']).difference({tripid}):
        tr_tripid_ind = data[data['tripid'] == i].index
        tr_tripid = [(data.loc[j, 'lon'], data.loc[j, 'lat']) for j in tr_tripid_ind]
        other_tr += [tr_tripid]
        other_tripid += [i]
    
    # calculate hausdorff distances
    for tr in other_tr:
        dist += [directed_hausdorff(queried, tr)[0]]
        #dist += [directed_hausdorff(k, queried)[0]]
    
    # find 
    tripid_dist = dict(zip(other_tripid, dist))
    return dict(sorted(tripid_dist.items(), key = lambda x: x[1])[:k])
	
# -----------------------------------------------------------------
# Cases
# -----------------------------------------------------------------

# Task 5 Case 1 Hausdorff distance

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(hsdf_most_sim_tr(100000, 20))

# Time and memory cost incurred by the query    
after_query()

# Task 5 Case 2 Hausdorff distance

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(hsdf_most_sim_tr(100000, 30))

# Time and memory cost incurred by the query    
after_query()

# Task 5 Case 3 Hausdorff distance

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(hsdf_most_sim_tr(100098, 20))

# Time and memory cost incurred by the query    
after_query()

# Task 5 Case 4 Hausdorff distance

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(hsdf_most_sim_tr(100098, 30))

# Time and memory cost incurred by the query    
after_query()

# -----------------------------------------------------------------
# Algorithm 5 DTW distance
# -----------------------------------------------------------------

from fastdtw import fastdtw
from haversine import haversine
import numpy as np
import pandas as pd

def great_circle(point1, point2):
    '''Calculate the great-circle distance between two lat-lon pairs.
       The inputs 'point1' and 'point2' are lon-lat pairs.
    '''
    lon1, lat1 = point1
    lon2, lat2 = point2
    return haversine((lat1, lon1), (lat2, lon2), unit='km')

def most_sim_tr(tripid, data=taxi_excerpt):
    '''Find a trajectory that is most similar to a queried trajectory.
       Note: 'tripid' is the one in 'data'
    '''
    min_dist = float('inf')
    most_similar_tr = None
    other_tr = []
    other_tripid = []
    
    # extract the queried trajectory by tripid
    queried_ind = data[data['tripid'] == tripid].index
    queried = [[data.loc[i, 'lon'], data.loc[i, 'lat']] for i in queried_ind]
    
    # extract the other trajectory by the other tripid's
    for i in set(taxi_excerpt['tripid']).difference({tripid}):
        tr_tripid_ind = data[data['tripid'] == i].index
        tr_tripid = [[data.loc[j, 'lon'], data.loc[j, 'lat']] for j in tr_tripid_ind]
        other_tr += [tr_tripid]
        other_tripid += [i]
    
    # sort out the trajectory of minimum DTW distance to the queried trajectory
    for tr in other_tr:
        distance, path = fastdtw(queried, tr, dist=great_circle)
        if distance < min_dist:
            min_dist = distance
            most_similar_tr = tr
    
    # return the tripid of the most similar trajectory, with its DTW distance to the
    # queried trajectory
    return other_tripid[other_tr.index(most_similar_tr)], min_dist

# -----------------------------------------------------------------
# Cases - The cases of DTW distance cannot be tested against ground truth, as PostgreSQL does not support DTW distance
# -----------------------------------------------------------------

# Task 5 Case 1 DTW distance (not realizable in PostgreSQL)

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(most_sim_tr(100000))

# Time and memory cost incurred by the query    
after_query()

# Task 5 Case 2 DTW distance (not realizable in PostgreSQL)

# Time and memory cost before query
time_before, memory_before = time.time(), psutil.Process().memory_info().rss / 1024**2

print(most_sim_tr(100098))

# Time and memory cost incurred by the query    
after_query()



# =================================================================
# Plot the results
# =================================================================

# An excerpt of 'results.xlsx'：
# Task  Case	 Algorithm	 precision recall  f1	     time     memory
# A	    1	 Linear scan	 1.000     1.000   1.000  277.877  479.403 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel('results.xlsx', header=0)

# mapping for other algorithm for each task
other_algorithm_map = {
    'A': 'R-tree',
    'B': 'Ball tree',
    'C': 'Ball tree',
    'D': 'kd tree',
    'E': 'Hausdorff'
}

tasks = df['Task'].unique()
cases = df['Case'].unique()
metrics_left = ['precision', 'recall', 'f1']
metrics_right = ['time', 'memory']

n_cases = len(cases)
bar_width = 0.2  # the width of the bars
gap_width = 0.1  # the width of the gaps between bar groups

# create subplots with one for each task
fig, axs = plt.subplots(len(tasks), 1, figsize=(8, 15), dpi=600, facecolor='white')

# iterate over each task
for i, task in enumerate(tasks):
    # filter data for current task
    df_task = df[df['Task'] == task]
    
    # get other algorithm for current task
    other_algorithm = other_algorithm_map[task]

    # add second y axis for right side metrics (time, memory)
    axs2 = axs[i].twinx()

    # iterate over each case
    for j, case in enumerate(cases):
        df_case = df_task[df_task['Case'] == case]
        x = np.arange(len(metrics_left) + len(metrics_right))  # the label locations
        offset = (bar_width * n_cases + gap_width) * (j - (n_cases - 1) / 2)

        # plot metrics for left axis (precision, recall, f1)
        rects1 = axs[i].bar(x[:len(metrics_left)] * (n_cases + gap_width) + offset - bar_width / 2, df_case[df_case['Algorithm'] == 'Linear scan'][metrics_left].values[0], bar_width, label=f'Case {case} Linear scan')
        rects2 = axs[i].bar(x[:len(metrics_left)] * (n_cases + gap_width) + offset + bar_width / 2, df_case[df_case['Algorithm'] == other_algorithm][metrics_left].values[0], bar_width, label=f'Case {case} {other_algorithm}')

        # plot time and memory on the second y axis
        rects3 = axs2.bar(x[len(metrics_left):] * (n_cases + gap_width) + offset - bar_width / 2, df_case[df_case['Algorithm'] == 'Linear scan'][metrics_right].values[0], bar_width, color=rects1[0].get_facecolor())
        rects4 = axs2.bar(x[len(metrics_left):] * (n_cases + gap_width) + offset + bar_width / 2, df_case[df_case['Algorithm'] == other_algorithm][metrics_right].values[0], bar_width, color=rects2[0].get_facecolor())

    axs[i].set_xlabel('Metrics')
    axs[i].set_ylabel('Precision; Recall; f1')    
    axs[i].set_title(f'Task {task}')
    axs[i].set_xticks(x * (n_cases + gap_width))
    axs[i].set_xticklabels(metrics_left + metrics_right)
    #axs[i].legend(loc='upper left', bbox_to_anchor=(1.07,1))
    axs[i].set_ylim(0, 1)  # scale for left y-axis

    axs2.set_ylabel('Time (sec); Memory (MB)')
    axs2.set_ylabel('Time (sec); Memory (MB)')
    axs2.set_ylim(0, df_task[metrics_right].max().max())  # scale for right y-axis individually for each task

plt.tight_layout()
plt.savefig('fig2.pdf')
plt.show()



# =================================================================
# End of file
# =================================================================