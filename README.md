![celestialink](images/banner.png)

# Boosting classification algorithms for mapping points in hyperspace

## SegmentedSVC
This works by chopping up the space and training a different SVC per segment.  This performs much faster than using a simple SVC without significant accuracy loss.  The results may also be better than the simple SVC for certain data sets.