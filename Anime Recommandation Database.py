#Name: sziaa
#Date: 08/23/2023
#Anime Recommendation Database
#Code from Kaggle
#https://www.kaggle.com/code/bastianbagas/starter-anime-recommendations-database-d0ac17ee-d
#import files anime.csv and rating.csv

#import for 3D axis'
from mpl_toolkits.mplot3d import Axes3D

#import for scaling to unit variance
from sklearn.preprocessing import StandardScaler

#import for plotting
import matplotlib.pyplot as plt 

#import for linear algebra
import numpy as np 

#import for creating directory structure
import os

#import for processing csv files
import pandas as pd


######Functions for creating Data Summaries of Results######

#Create histograms/barcharts of csv. columns data
def hist_per_col(df,nGraph,nGraphRow):

    #number of unique values in each column
    nunique = df.nunique()

    #pick columns that have between 1 and 50 unique values
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]

    #number of rows in the filtered df
    nRow, nCol = df.shape

    #List filtered column names
    columnNames = list(df)

    #calculate number of rows needed for subplots
    nPlotRows = (nCol + nGraphRow - 1)//nGraphRow
 
    #create plot from matplotlib.pyplot
    plt.figure(num=None, figsize = (6 * nGraphRow, 8 * nPlotRows), dpi= 80, facecolor = 'w', edgecolor = 'k')

    #loop over each column and make subplots
    for i in range(min(nCol,nGraph)):
        
        #create subplot for ith column
        plt.subplot(nPlotRows, nGraphRow, i+1)

        #extract data from the ith column
        colDf = df.iloc[:,1]

        #if column has non numeric data plot a bar chart, otherwise a historgram
        if (not np.issubdtype(type(colDf.iloc[0]), np.number)):
            valueCount = colDf.value_counts()
            valueCount.plot.bar()
        else:
            colDf.hist()

        #set labels for subplots
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column{i})')

    #adjust layout and display plots
    plt.tight_layout(pad=1.0,w_pad = 1.0, h_pad = 1.0)
    plt.show()


#Create correlation matrix of given data
def plot_Corr_Matrix(df,graphWidth):

    #Store name of data frame 
    filename = df.dataframeName

    #drop empty columns
    df = df.dropna('columns')

    #filter to keeps columns that have more than 1 unique value only
    df = df[[col for col in df if df[col].nunique() > 1]]

    #make sure there are at lest 2 columns remainig in the filtered data
    if df.shape[1] < 2:
        print(f'No Correlation Plots Available:The number of non-NaN or constant columns ({df.shape[1]}) is less than 2!')
        return

    #Calcuate Correlation Matrix of Data Frame
    corr =df.corr()

    #create figure for matrix
    plt.figure(num = None, figsize = (graphWidth, graphWidth),dpi =80,facecolor='w', edgecolor='k')

    #display correlation matrix, colour coded
    corrMatrix = plt.matshow(corr,fignum = 1)

    #set x-axis labels to column names and rotate
    plt.xticks(range(len(corr.column)), corr.columns,rotation=90)

    #set y-axis labels to column names
    plt.yticks(range(len(corr.column)), corr.columns)

    #Move x-axis lables to the bottom
    plt.gca().xaxis.tick_bottom()

    #Add colour bar to the plot to show a coorelation scale
    plt.colorbar(corrMatrix)

    #Create Matrix Title
    plt.title(f'Corelation Matrix for {filename}', fontsize = 15)

    #Display
    plt.show()


#Create Scatter and Density Plots
def plot_Scatter(df, plotSize, textSize):

    #Filter columns to only have numerical data types
    df = df.select_dtypes(include=[np.number])

    #filter for non empty value columns
    df = df.dropna('columns')

    #only keep columns that have more than 1 unique value
    df = df[[col for col in df if df[col].nunique() > 1]]

    #list of column names in data set
    columnNames = list(df)
    
    #reduce the number of columns for matrix inversion of kernel density plots
    if len(columnNames)>10:
        columnNames = columnNames[:10]

    #Keep only selected columns in Df
    df = df[columnNames]

    #Gerenate Scatter/Density Plots
    ax = pd.plotting.scatter_matrix(df,alpha=0.75,figsize = [plotSize,plotSize], diagonal = 'kde')

    #Calculate Correlations Coefficients for DF
    corrs = df.corr().values

    #Add correlation coefficients into matrix as lables
    for i, j in zip(*plt.np.trie_indices_from(ax, k = 1)):
        ax[i,j].annotate('Corr. coef = %.3f'% corrs[i,j],(0.8,0.2),xycoords = 'axes fraction', ha ='center', va = 'center', size = testSize)

    #Create Plot Title
    plt.subtitle('Scatter and Density Plot')

    #Display
    plt.show()

#Check anime.csv data set
nRowsRead = 1000
df1 = pd.read_csv('anime.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'anime.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
    
#Display the histograms/bargraphs of the data sets
hist_per_col(df1, 10, 5)




