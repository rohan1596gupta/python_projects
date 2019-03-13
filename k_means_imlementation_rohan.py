# -*- coding: utf-8 -*-
"""

K-means clustering algorithm for given dataset (image vector)
coded by : Rohan Kumar
dated : 14/03/2019

"""

# importing relevant libraries
import csv
import numpy as np
import matplotlib.pyplot as pl

def get_centroids(k, data):
    ''' getting centroids randomly from the dataset we have generated '''
    centroids = []
    for i in range(0, k):
        cluster_index = np.random.randint(len(data), size=1)
        centroids.append(data[cluster_index])
    return np.array(centroids)

def k_means(data, centroids):
    ''' k-means algorithm implementation'''
    total_points = len(data)        # number of features in data
    k = len(centroids)              # number of centroids
    
    for idx_data in range(0, total_points):
        distance = {}
        for idx_centroid in range(0, k):
            distance[idx_centroid] = np.linalg.norm(data[idx_data] - centroids[idx_centroid])       # euclidean distance b/w two data points
        
        idx_min = min(distance, key=distance.get)       # getting index for nearest centroid of data
        centroids[idx_min] = np.array(data[idx_data] + centroids[idx_min])/2        # mean of old and new centroid
            
    return centroids

def create_csv_datafile():
    #Extracting first 200 lines of dataset(representing digit'0') and loading it into seperate csv file
    with open('/Users/rohankumar/Desktop/classes 2019/Machine Leanrning/DigitsBasicRoutines/mfeat-pix.txt', 'r') as in_file1:
        head = [next(in_file1) for x in range(200)]     # getting first 200 lines
        stripped1 = (line.strip() for line in head)     # stripping unwanted newline and , separator
        lines1 = (line.strip('\n').split('  ') for line in stripped1 if line)
        with open('/Users/rohankumar/Desktop/classes 2019/Machine Leanrning/DigitsBasicRoutines/mfeat-pix_sample.csv', 'w', newline='') as out_file1:
            writer = csv.writer(out_file1)
            writer.writerows(lines1)
    return '/Users/rohankumar/Desktop/classes 2019/Machine Leanrning/DigitsBasicRoutines/mfeat-pix_sample.csv'
    
def main():
    width, height = 16, 15  # pixel dimension
    k = 6   # number of clusters
        
    filename = create_csv_datafile()       # creating csv file from the raw .pix data (file)
    
    ''' converting data into vectors '''
    datafile = open(filename, 'rt')
    data = np.loadtxt(datafile, delimiter=',', dtype=int)
    datafile.close()

    centroids = get_centroids(k, data)      # getting random centroids from data
    new_centroids = k_means(data, centroids)  # newly generated centroids for clusters
    print('centroids', len(centroids), '\nnew_centroids', (new_centroids))
    
    list_new_centroids = list(new_centroids)
    print('list new centroids', list_new_centroids)
    new_data = np.zeros((len(centroids), width, height), dtype=int)     # reshaped np array for plotting images vector
    print('\nnew data\n', (new_data))

    fig = pl.figure()
    cluster_index = 1
    for i in range(len(list_new_centroids)):
        #if i%22 == 0 and cluster_index <= 9:
        new_data[i] = np.reshape(list_new_centroids[i], (width, height))
        ax = pl.subplot(1, k, cluster_index)
        cluster_index += 1
        ax.set_title('Cluster '+str(i+1), fontsize=10)
        ax.imshow(new_data[i], cmap='gray_r')

    fig.set_tight_layout(True)      # automatically adjusts subplot to fits into the figure area
    pl.show()

main()