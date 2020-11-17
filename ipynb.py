#!/usr/bin/env python
# coding: utf-8

# Geolocate

# In[282]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pinfo', 'plt.plot')


# # Creating Scatter Plots

# In[283]:


x = np.array([1,2,3,4,5,6,7,8])
y = x

plt.figure()
plt.scatter(x, y) 


# In[287]:


x = np.array([1,2,3,4,5,6,7,8,9,10])
y = x

# Let's create a list of colors for each point to have:
# ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green' , 'green' , orange']
colors = ['green']*(len(x)-1)
colors.append('orange')

plt.figure()

# plot the point with size 100 and chosen colors
plt.scatter(x, y, s=100, c=colors)


# In[289]:


# convert lists into a list of tuples
zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])

print(list(zip_generator))
# the above prints:
# [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]

zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
# The below single star * unpacks a collection into positional arguments
print(*zip_generator)
# the above prints:
# (1, 6) (2, 7) (3, 8) (4, 9) (5, 10)


# In[290]:


# use zip to convert 5 tuples with 2 elements each to 2 tuples with 5 elements each
print(list(zip((1, 6), (2, 7), (3, 8), (4, 9), (5, 10))))
# the above prints:
# [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)]


zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
# let's turn the data back into 2 lists
x, y = zip(*zip_generator) # This is like calling zip((1, 6), (2, 7), (3, 8), (4, 9), (5, 10))
print(x)
print(y)
# the above prints:
# (1, 2, 3, 4, 5)
# (6, 7, 8, 9, 10)


# In[291]:


plt.figure()
# plot a data series 'Inactive Learner' in red using the first two elements of x and y
plt.scatter(x[:2], y[:2], s=100, c='red', label='Inactive Learner')

# plot a second data series 'Active Learner' in yellow using the last three elements of x and y 
plt.scatter(x[2:], y[2:], s=100, c='yellow', label='Active Learner')


# In[238]:


# add a label to the x axis
plt.xlabel('The number of online courses the learner accomplished')

# add a label to the y axis
plt.ylabel('The grade of the learners')


# In[292]:


# add a title
plt.title('Relationship between number of courses and grades')

# add a legend (uses the labels from plt.scatter)
plt.legend()

# add the legend to loc=4 (the lower right hand corner), also gets rid of the frame and adds a title
plt.legend(loc=4, frameon=False, title='Legend')


# # Creating Line Plots

# In[293]:


linear_data = np.array([1,2,3,4,5,6,7,8])
exponential_data = linear_data**2

plt.figure()
# plot the linear data and the exponential data
plt.plot(linear_data, '-o', exponential_data, '-o')


# In[241]:


# plot another series with a dashed red line
plt.plot([22,44,55], '--r')


# In[294]:


plt.xlabel('Number of AI courses accomplished')
plt.ylabel('Progress in a Data Science Career')
plt.title('Impact of taking Coursera Courses on a Data Science Career')
# add a legend with legend entries (because we didn't have labels when we plotted the data series)
plt.legend(['Basic Courses', 'Advanced Courses', 'other topics'])


# In[295]:


# fill the area between the linear data and exponential data
plt.gca().fill_between(range(len(linear_data)), 
                       linear_data, exponential_data, 
                       facecolor='Blue', 
                       alpha=0.20)


# # Creating Bar Charts

# In[296]:


plt.figure()
xvals = range(len(linear_data))
plt.bar(xvals, linear_data, width = 0.4)


# In[245]:


new_xvals = []

# plot another set of bars, adjusting the new xvals to make up for the first set of bars plotted
for item in xvals:
    new_xvals.append(item+0.3)

plt.bar(new_xvals, exponential_data, width = 0.4 ,color='Yellow')


# In[297]:


from random import randint
linear_err = [randint(0,15) for x in range(len(linear_data))] 

# This will plot a new set of bars with errorbars using the list of random error values
plt.bar(xvals, linear_data, width = 0.3, yerr=linear_err)


# In[298]:


# create stacked bar charts
plt.figure()
xvals = range(len(linear_data))
plt.bar(xvals, linear_data, width = 0.3, color='b')
plt.bar(xvals, exponential_data, width = 0.3, bottom=linear_data, color='r')


# In[248]:


# create horizontal bar charts
plt.figure()
xvals = range(len(linear_data))
plt.barh(xvals, linear_data, height = 0.3, color='b')
plt.barh(xvals, exponential_data, height = 0.3, left=linear_data, color='r')


# # Creating Subplots

# In[300]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('pinfo', 'plt.subplot')


# In[301]:


plt.figure()
# subplot with 1 row, 2 columns, and current axis is 1st subplot axes
plt.subplot(1, 2, 1)

linear_data = np.array([1,2,3,4,5,6,7,8])

plt.plot(linear_data, '-o')


# In[251]:


exponential_data = linear_data**2 

# subplot with 1 row, 2 columns, and current axis is 2nd subplot axes
plt.subplot(1, 2, 2)
plt.plot(exponential_data, '-o')


# In[302]:


# plot exponential data on 1st subplot axes
plt.subplot(1, 2, 1)
plt.plot(exponential_data, '-x')


# In[303]:


plt.figure()
ax1 = plt.subplot(1, 2, 1)
plt.plot(linear_data, '-o')
# pass sharey=ax1 to ensure the two subplots share the same y axis
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
plt.plot(exponential_data, '-x')


# In[313]:


plt.figure()
# the right hand side is equivalent shorthand syntax
plt.subplot(1,2,1) == plt.subplot(111)


# In[309]:


# create a 3x3 grid of subplots
fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3, 3, sharex=True, sharey=True)
# plot the linear_data on the 5th subplot axes 
ax5.plot(linear_data, '-')


# In[314]:


# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

#A figure with just one subplot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('A single plot')


# In[259]:


# Stacking subplots in one direction

# vertically
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(x, y)
axs[1].plot(x, -y)

# horizontally
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Horizontally stacked subplots')
ax1.plot(x, y)
ax2.plot(x, -y)


# In[315]:


# Stacking subplots in two directions
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 0].set_title('Axis [0,0]')
axs[0, 1].plot(x, y, 'tab:orange')
axs[0, 1].set_title('Axis [0,1]')
axs[1, 0].plot(x, -y, 'tab:green')
axs[1, 0].set_title('Axis [1,0]')
axs[1, 1].plot(x, -y, 'tab:red')
axs[1, 1].set_title('Axis [1,1]')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()


# # Creating Histograms

# In[323]:


# create 2x2 grid of axis subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1,ax2,ax3,ax4]

# draw n = 10, 100, 1000, and 10000 samples from the normal distribution and plot corresponding histograms
for n in range(0,len(axs)):
    sample_size = 10**(n+1)
    sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
    axs[n].hist(sample)
    axs[n].set_title('n={}'.format(sample_size))


# In[262]:


# repeat with number of bins set to 200
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1,ax2,ax3,ax4]

for n in range(0,len(axs)):
    sample_size = 10**(n+1)
    sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
    axs[n].hist(sample, bins=200)
    axs[n].set_title('n={}'.format(sample_size))


# In[324]:


plt.figure()
Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
plt.scatter(X,Y)


# In[320]:


# use gridspec to partition the figure into subplots
import matplotlib.gridspec as gridspec

plt.figure()
gspec = gridspec.GridSpec(3, 3)

top_histogram = plt.subplot(gspec[0, 1:])
side_histogram = plt.subplot(gspec[1:, 0])
lower_right = plt.subplot(gspec[1:, 1:])


# In[325]:


Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
lower_right.scatter(X, Y)
top_histogram.hist(X, bins=100)
s = side_histogram.hist(Y, bins=100, orientation='horizontal')


# In[326]:


# clear the histograms and plot normed histograms
top_histogram.clear()
top_histogram.hist(X, bins=100, normed=True)
side_histogram.clear()
side_histogram.hist(Y, bins=100, orientation='horizontal', normed=True)
# flip the side histogram's x axis
side_histogram.invert_xaxis()


# In[327]:


# change axes limits
for ax in [top_histogram, lower_right]:
    ax.set_xlim(0, 1)
for ax in [side_histogram, lower_right]:
    ax.set_ylim(-5, 5)


# # Create Box and Whisker Plots

# In[328]:


import pandas as pd
normal_sample = np.random.normal(loc=0.0, scale=1.0, size=10000)
random_sample = np.random.random(size=10000)
gamma_sample = np.random.gamma(2, size=10000)

df = pd.DataFrame({'normal': normal_sample, 
                   'random': random_sample, 
                   'gamma': gamma_sample})


# In[329]:


df.describe()


# In[330]:


plt.figure()
# create a boxplot of the normal data, assign the output to a variable to supress output
_ = plt.boxplot(df['normal'], whis='range')


# In[271]:


# clear the current figure
plt.clf()
# plot boxplots for all three of df's columns
_ = plt.boxplot([ df['normal'], df['random'], df['gamma'] ], whis='range')


# In[331]:


plt.figure()
_ = plt.hist(df['gamma'], bins=100)


# In[273]:


import mpl_toolkits.axes_grid1.inset_locator as mpl_il

plt.figure()
plt.boxplot([ df['normal'], df['random'], df['gamma'] ], whis='range')
# overlay axis on top of another 
ax2 = mpl_il.inset_axes(plt.gca(), width='60%', height='40%', loc=2)
ax2.hist(df['gamma'], bins=100)
ax2.margins(x=0.5)


# In[332]:


# switch the y axis ticks for ax2 to the right side
ax2.yaxis.tick_right()


# In[333]:


# if `whis` argument isn't passed, boxplot defaults to showing 1.5*interquartile (IQR) whiskers with outliers
plt.figure()
_ = plt.boxplot([ df['normal'], df['random'], df['gamma'] ] )


# # Create Heatmaps

# In[334]:


plt.figure()

Y = np.random.normal(loc=0.0, scale=1.0, size=15000)
X = np.random.random(size=15000)
_ = plt.hist2d(X, Y, bins=30)


# In[277]:


plt.figure()
_ = plt.hist2d(X, Y, bins=100)


# In[335]:


# add a colorbar legend
plt.colorbar()


# # Create Animations

# In[336]:


import matplotlib.animation as animation

n = 100
x = np.random.randn(n)


# In[337]:


# create the function that will do the plotting, where curr is the current frame
def update(curr):
    # check if animation is at the last frame, and if so, stop the animation a
    if curr == n: 
        a.event_source.stop()
    plt.cla()
    bins = np.arange(-4, 4, 0.5)
    plt.hist(x[:curr], bins=bins)
    plt.axis([-4,4,0,30])
    plt.gca().set_title('Sampling the Normal Distribution')
    plt.gca().set_ylabel('Frequency')
    plt.gca().set_xlabel('Value')
    plt.annotate('n = {}'.format(curr), [3,27])


# In[338]:


fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval=100)


# 

# # Congratulations! You are now able to create amazing plots and visualize any type of data in your own Machine Learning Projects!

# In[ ]:




