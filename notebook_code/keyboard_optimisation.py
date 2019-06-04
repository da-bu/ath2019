import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# this is just an example of what an optimisation landscape could look like:
def plot_example_optimisation_landscape():
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm

	sns.set_style('white')
	plt.figure(figsize=(14, 7))
	ax = plt.subplot(111, projection='3d')


	stepsX = 16 * 32
	stepsY = 16 * 32
	xi = np.linspace(-10, 10, stepsX, True)
	yi = np.linspace(-10, 10, stepsY, True)
	X, Y = np.meshgrid(xi, yi)
	R = np.sqrt(X**2 + Y**2)
	#Z = np.sin(2*R) + np.cos(2*R) - X*np.sin(.5*Y)
	#Z += np.sqrt((X-1)**2 +(Y-1)**2)

	from scipy.stats import multivariate_normal

	def place_gaussian(X, Y, mean, cov):
	    return multivariate_normal.pdf(np.hstack((X.reshape(-1,1),Y.reshape(-1,1))), mean, cov).reshape(X.shape)

	Z = 0.001*(np.sin(2*R) + np.cos(2*R) + X*np.sin(.25*Y))
	#Z += 0.005*np.sqrt((X-10)**2 +(Y-1)**2)
	Z += 2*place_gaussian(X, Y, mean=[5,5], cov=5*np.eye(2))
	Z -= 4*place_gaussian(X, Y, mean=[8,-8], cov=15*np.eye(2))
	Z += 12*place_gaussian(X, Y, mean=[-8,8], cov=15*np.eye(2))

	CS = ax.plot_surface(X, Y, Z, cmap=cm.BrBG_r, linewidth=0, antialiased=True)

	plt.tight_layout()
	plt.show()
	#plt.savefig('optimisation_landscape.png', dpi=100)


# this is just another example of what an optimisation landscape could look like:
def plot_example_optimisation_landscape2():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    sns.set_style('white')
    plt.figure(figsize=(14, 7))
    ax = plt.subplot(111, projection='3d')
    ax.view_init(elev=55, azim=-60)


    stepsX = 16 * 32
    stepsY = 16 * 32
    xi = np.linspace(-10, 10, stepsX, True)
    yi = np.linspace(-10, 10, stepsY, True)
    X, Y = np.meshgrid(xi, yi)
    R = np.sqrt(X**2 + Y**2)
    
    from scipy.stats import multivariate_normal

    def place_gaussian(X, Y, mean, cov):
        return multivariate_normal.pdf(np.hstack((X.reshape(-1,1),Y.reshape(-1,1))), mean, cov).reshape(X.shape)
    
    Z = 0
    for i in xrange(200):
        Z += (np.random.rand()*30-15)*place_gaussian(X, Y, mean=[np.random.rand()*30-15, np.random.rand()*30-15], cov=np.eye(2)*(1+np.random.rand()*1))
  

    CS = ax.plot_surface(X, Y, Z, cmap=cm.BrBG_r, linewidth=0, antialiased=True)

    plt.tight_layout()
    plt.show()
    plt.savefig('optimisation_landscape_2.png', dpi=100)





# Create the key locations (x, y per key) - similar to the Android keyboard:
def create_key_locations(key_w=1, key_h=1.5):
    key_locations = []
    # first row locations:
    for i in range(10):
        x = i*key_w
        y = 0
        key_locations.append([x,y])
    # second row locations:
    for i in range(9):
        x = i*key_w + key_w*0.5
        y = key_h
        key_locations.append([x,y])
    # third row locations:
    for i in range(7):
        x = i*key_w + key_w*1.5
        y = 2*key_h
        key_locations.append([x,y])   
    return np.array(key_locations)

# Visualises a keyboard layout, given the characters in order and the corresponding key locations:
def plot_keyboard(layout, key_locations, title='Keyboard'):
    plt.figure()
    ax = plt.subplot(111)
    ax.invert_yaxis()
    plt.axis('off')
    plt.scatter(key_locations[:,0], key_locations[:,1], c='k', s=0)
    for i, character in enumerate(layout):
        ax.annotate(character, key_locations[i], horizontalalignment='center', verticalalignment='center', fontsize=22)    
    plt.title(title+'\n', fontsize=22)
    plt.tight_layout()
    plt.show()

# Measures the euclidean distance between two key locations:
def distance(key_location, key_location2):
    x,y = key_location
    x2,y2 = key_location2
    return np.sqrt((x-x2)**2 + (y-y2)**2)


# Computes the typing speed in words per minute (WPM), given the mean time interval between two key presses:
def wpm(mean_inter_key_time):
    return (1.0/mean_inter_key_time*60)/5