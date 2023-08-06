/*
Project: Spatial Algorithms on Spatio Temporal Taxi Trajectories
Author: Longpeng Xu (https://github.com/mattxu98/)
Date: 20230525
*/
-- ---------------------------------------------------
-- Database Construction
-- ---------------------------------------------------

-- taxi_excerpt

CREATE TABLE public.taxi_excerpt
(
    tripid integer,
    pointid integer NOT NULL,
    timestamp1 timestamp with time zone,
    stand real,
    lon double precision,
    lat double precision,
    PRIMARY KEY (pointid)
);

COPY taxi_excerpt (tripid, pointid, timestamp1, stand, lon, lat)
FROM 'D:/Current/INFS7205/7205project/taxi_excerpt.csv'
DELIMITER ',' CSV HEADER;

-- taxi_sql

CREATE TABLE taxi_sql (
  tripid serial PRIMARY KEY,
  trajectory geometry(LineString, 4326));
  
COPY taxi_sql (tripid, trajectory)
FROM 'D:/Current/INFS7205/7205project/taxi_sql.csv'
DELIMITER ',' CSV HEADER;

-- ---------------------------------------------------
-- Database Manipulation
-- ---------------------------------------------------

-- taxi_excerpt

ALTER TABLE taxi_excerpt ADD COLUMN location geometry(Point, 4326);
UPDATE taxi_excerpt SET location = ST_SetSRID(ST_MakePoint(lon, lat), 4326);

-- taxi_sql: No database manipulation applied

-- ---------------------------------------------------
-- Tasks
-- ---------------------------------------------------

-- Task A Case 1

SELECT pointid
FROM taxi_excerpt
WHERE lon >= -8.59 AND lon <= -8.54
  AND lat >= 41.13 AND lat <= 41.17
  AND timestamp1 >= '2013-07-01 10:00:00' AND timestamp1 <= '2013-07-01 10:05:00';

-- Task A Case 2

SELECT pointid
FROM taxi_excerpt
WHERE lon >= -8.59 AND lon <= -8.54
  AND lat >= 41.13 AND lat <= 41.17
  AND timestamp1 >= '2013-07-01 11:10:00' AND timestamp1 <= '2013-07-01 11:15:00';

-- Task A Case 3

SELECT pointid
FROM taxi_excerpt
WHERE lon >= -8.65 AND lon <= -8.58
  AND lat >= 41.15 AND lat <= 41.20
  AND timestamp1 >= '2013-07-01 10:00:00' AND timestamp1 <= '2013-07-01 10:05:00';

-- Task A Case 4

SELECT pointid
FROM taxi_excerpt
WHERE lon >= -8.65 AND lon <= -8.58
  AND lat >= 41.15 AND lat <= 41.20
  AND timestamp1 >= '2013-07-01 11:10:00' AND timestamp1 <= '2013-07-01 11:15:00';

-- Task B Case 1

WITH reference_points AS (
  SELECT pointid, ST_MakePoint(lon, lat) AS ref_point
  FROM taxi_excerpt
  WHERE tripid <> 100000
),
trajectory_points AS (
  SELECT ST_MakePoint(ST_X((dp).geom), ST_Y((dp).geom)) AS point
  FROM (SELECT ST_DumpPoints(trajectory) AS dp
		FROM taxi_sql
		WHERE tripid = 100000) AS sub
),
distances AS (
SELECT rp.pointid, ST_DistanceSphere(rp.ref_point, tp.point) AS distance
FROM reference_points rp
CROSS JOIN trajectory_points tp
),
min_distances AS (
  SELECT pointid, MIN(distance) AS min_distance
  FROM distances
  GROUP BY pointid
)
SELECT pointid, min_distance
FROM min_distances
ORDER BY min_distance
LIMIT 20;

-- Task B Case 2

WITH reference_points AS (
  SELECT pointid, ST_MakePoint(lon, lat) AS ref_point
  FROM taxi_excerpt
  WHERE tripid <> 100000
),
trajectory_points AS (
  SELECT ST_MakePoint(ST_X((dp).geom), ST_Y((dp).geom)) AS point
  FROM (SELECT ST_DumpPoints(trajectory) AS dp
		FROM taxi_sql
		WHERE tripid = 100000) AS sub
),
distances AS (
SELECT rp.pointid, ST_DistanceSphere(rp.ref_point, tp.point) AS distance
FROM reference_points rp
CROSS JOIN trajectory_points tp
),
min_distances AS (
  SELECT pointid, MIN(distance) AS min_distance
  FROM distances
  GROUP BY pointid
)
SELECT pointid, min_distance
FROM min_distances
ORDER BY min_distance
LIMIT 40;

-- Task B Case 3

WITH reference_points AS (
  SELECT pointid, ST_MakePoint(lon, lat) AS ref_point
  FROM taxi_excerpt
  WHERE tripid <> 100035
),
trajectory_points AS (
  SELECT ST_MakePoint(ST_X((dp).geom), ST_Y((dp).geom)) AS point
  FROM (SELECT ST_DumpPoints(trajectory) AS dp
		FROM taxi_sql
		WHERE tripid = 100035) AS sub
),
distances AS (
SELECT rp.pointid, ST_DistanceSphere(rp.ref_point, tp.point) AS distance
FROM reference_points rp
CROSS JOIN trajectory_points tp
),
min_distances AS (
  SELECT pointid, MIN(distance) AS min_distance
  FROM distances
  GROUP BY pointid
)
SELECT pointid, min_distance
FROM min_distances
ORDER BY min_distance
LIMIT 20;

-- Task B Case 4

WITH reference_points AS (
  SELECT pointid, ST_MakePoint(lon, lat) AS ref_point
  FROM taxi_excerpt
  WHERE tripid <> 100035
),
trajectory_points AS (
  SELECT ST_MakePoint(ST_X((dp).geom), ST_Y((dp).geom)) AS point
  FROM (SELECT ST_DumpPoints(trajectory) AS dp
		FROM taxi_sql
		WHERE tripid = 100035) AS sub
),
distances AS (
SELECT rp.pointid, ST_DistanceSphere(rp.ref_point, tp.point) AS distance
FROM reference_points rp
CROSS JOIN trajectory_points tp
),
min_distances AS (
  SELECT pointid, MIN(distance) AS min_distance
  FROM distances
  GROUP BY pointid
)
SELECT pointid, min_distance
FROM min_distances
ORDER BY min_distance
LIMIT 40;

-- Task C Case 1

WITH reference_points AS (
  SELECT pointid, ST_MakePoint(lon, lat) AS ref_point
  FROM taxi_excerpt
  WHERE tripid <> 100000
),
trajectory_points AS (
  SELECT ST_MakePoint(ST_X((dp).geom), ST_Y((dp).geom)) AS point
  FROM (SELECT ST_DumpPoints(trajectory) AS dp
		FROM taxi_sql
		WHERE tripid = 100000) AS sub
),
distances AS (
SELECT rp.pointid, ST_DistanceSphere(rp.ref_point, tp.point) AS distance
FROM reference_points rp
CROSS JOIN trajectory_points tp
),
min_distances AS (
  SELECT pointid, MIN(distance) AS min_distance
  FROM distances
  GROUP BY pointid
)
SELECT pointid, min_distance
FROM min_distances
WHERE min_distance <= 80
ORDER BY min_distance;

-- Task C Case 2

WITH reference_points AS (
  SELECT pointid, ST_MakePoint(lon, lat) AS ref_point
  FROM taxi_excerpt
  WHERE tripid <> 100000
),
trajectory_points AS (
  SELECT ST_MakePoint(ST_X((dp).geom), ST_Y((dp).geom)) AS point
  FROM (SELECT ST_DumpPoints(trajectory) AS dp
		FROM taxi_sql
		WHERE tripid = 100000) AS sub
),
distances AS (
SELECT rp.pointid, ST_DistanceSphere(rp.ref_point, tp.point) AS distance
FROM reference_points rp
CROSS JOIN trajectory_points tp
),
min_distances AS (
  SELECT pointid, MIN(distance) AS min_distance
  FROM distances
  GROUP BY pointid
)
SELECT pointid, min_distance
FROM min_distances
WHERE min_distance <= 110
ORDER BY min_distance;

-- Task C Case 3

WITH reference_points AS (
  SELECT pointid, ST_MakePoint(lon, lat) AS ref_point
  FROM taxi_excerpt
  WHERE tripid <> 100024
),
trajectory_points AS (
  SELECT ST_MakePoint(ST_X((dp).geom), ST_Y((dp).geom)) AS point
  FROM (SELECT ST_DumpPoints(trajectory) AS dp
		FROM taxi_sql
		WHERE tripid = 100024) AS sub
),
distances AS (
SELECT rp.pointid, ST_DistanceSphere(rp.ref_point, tp.point) AS distance
FROM reference_points rp
CROSS JOIN trajectory_points tp
),
min_distances AS (
  SELECT pointid, MIN(distance) AS min_distance
  FROM distances
  GROUP BY pointid
)
SELECT pointid, min_distance
FROM min_distances
WHERE min_distance <= 80
ORDER BY min_distance;

-- Task C Case 4

WITH reference_points AS (
  SELECT pointid, ST_MakePoint(lon, lat) AS ref_point
  FROM taxi_excerpt
  WHERE tripid <> 100024
),
trajectory_points AS (
  SELECT ST_MakePoint(ST_X((dp).geom), ST_Y((dp).geom)) AS point
  FROM (SELECT ST_DumpPoints(trajectory) AS dp
		FROM taxi_sql
		WHERE tripid = 100024) AS sub
),
distances AS (
SELECT rp.pointid, ST_DistanceSphere(rp.ref_point, tp.point) AS distance
FROM reference_points rp
CROSS JOIN trajectory_points tp
),
min_distances AS (
  SELECT pointid, MIN(distance) AS min_distance
  FROM distances
  GROUP BY pointid
)
SELECT pointid, min_distance
FROM min_distances
WHERE min_distance <= 110
ORDER BY min_distance;

-- Task D Case 1

WITH center_point AS(
    SELECT ST_SetSRID(ST_MakePoint(lon, lat), 4326) AS center
	FROM taxi_excerpt
	WHERE pointid = 100001000
)
SELECT pointid
FROM taxi_excerpt, center_point
WHERE ST_DWithin(
    ST_SetSRID(ST_MakePoint(lon, lat), 4326), center_point.center,
    200 / (6371.009 * 1000) * 180 / pi() 
);

-- Task D Case 2

WITH center_point AS(
    SELECT ST_SetSRID(ST_MakePoint(lon, lat), 4326) AS center
	FROM taxi_excerpt
	WHERE pointid = 100001000
)
SELECT pointid
FROM taxi_excerpt, center_point
WHERE ST_DWithin(
    ST_SetSRID(ST_MakePoint(lon, lat), 4326), center_point.center,
    500 / (6371.009 * 1000) * 180 / pi() 
);

-- Task D Case 3

WITH center_point AS(
    SELECT ST_SetSRID(ST_MakePoint(lon, lat), 4326) AS center
	FROM taxi_excerpt
	WHERE pointid = 100048000
)
SELECT pointid
FROM taxi_excerpt, center_point
WHERE ST_DWithin(
    ST_SetSRID(ST_MakePoint(lon, lat), 4326), center_point.center,
    200 / (6371.009 * 1000) * 180 / pi() 
);

-- Task D Case 4

WITH center_point AS(
    SELECT ST_SetSRID(ST_MakePoint(lon, lat), 4326) AS center
	FROM taxi_excerpt
	WHERE pointid = 100048000
)
SELECT pointid
FROM taxi_excerpt, center_point
WHERE ST_DWithin(
    ST_SetSRID(ST_MakePoint(lon, lat), 4326), center_point.center,
    500 / (6371.009 * 1000) * 180 / pi() 
);

-- Task E Case 1 Hausdorff Distance

SELECT t2.tripid, ST_HausdorffDistance(t1.trajectory::geometry, t2.trajectory::geometry) AS distance
FROM taxi_sql t1, taxi_sql t2
WHERE t1.tripid = 100000 AND t1.tripid <> t2.tripid
ORDER BY distance ASC
LIMIT 20;

-- Task E Case 2 Hausdorff Distance

SELECT t2.tripid, ST_HausdorffDistance(t1.trajectory::geometry, t2.trajectory::geometry) AS distance
FROM taxi_sql t1, taxi_sql t2
WHERE t1.tripid = 100000 AND t1.tripid <> t2.tripid
ORDER BY distance ASC
LIMIT 30;

-- Task E Case 3 Hausdorff Distance

SELECT t2.tripid, ST_HausdorffDistance(t1.trajectory::geometry, t2.trajectory::geometry) AS distance
FROM taxi_sql t1, taxi_sql t2
WHERE t1.tripid = 100098 AND t1.tripid <> t2.tripid
ORDER BY distance ASC
LIMIT 20;

-- Task E Case 4 Hausdorff Distance

SELECT t2.tripid, ST_HausdorffDistance(t1.trajectory::geometry, t2.trajectory::geometry) AS distance
FROM taxi_sql t1, taxi_sql t2
WHERE t1.tripid = 100098 AND t1.tripid <> t2.tripid
ORDER BY distance ASC
LIMIT 30;


