'''
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
fig, ax =plt.subplots()
lc=LineCollection([[(0,0),(1,1)],[(-5,0),(5,0),(5,-4)]])

ax.add_collection(lc)
plt.show()
'''

import math
import matplotlib.pyplot as plt
import numpy as np
import imageio
import sys
from math import sqrt
from bresenham import bresenham
''' This is a simple RRT generator'''
def main():
    q_init = (50,50)          # Initial configuration
    nodes = [q_init]          # List of nodes
    K = 0                     # Number of vertices in nodes
    D = 100                 # Domain
    delta = 1                 # Incremental distance
    itr = 100  
    np.random.seed(100)               # Iterations
    for K in range (itr):
        q_rand = (np.random.rand()*D, np.random.rand()*D) 

        min_dist = math.inf              # Set infinity value for minmum distance
        
        for i in range(len(nodes)):
            new_dist = sqrt(pow((nodes[i][0] - q_rand[0]),2) + pow((nodes[i][1] - q_rand[1]),2))

            if new_dist < min_dist:
                min_dist = new_dist

                min_idx = i

        q_near = nodes[min_idx]               # finding nearest node
        
        q_vector = (q_rand[0]-q_near[0],q_rand[1]-q_near[1])       # q_rand to q_near in vector form

        q_V = np.array(q_vector)
        q_N = np.array(q_near)

        mag = np.sqrt(np.dot(q_V,q_V))          # magnitude of vector
        unit_vector = q_V/mag
        q_new = q_N + (delta) * (unit_vector)         # Move to new node in direction of q vector by step detla
        '''
        print(q_near)
        print(q_rand)
        unit_vector = np.array(q_near[0]-np.array(q_rand[0]), q_near[1]-np.array(q_rand[1]))/np.array(np.sqrt(q_near,q_rand))
        q_new = np.array(q_near) + (delta * unit_vector)
        '''
        nodes.append(q_new)                     # Add new nodes to list

        # Plot
        new_x = (q_near[0],q_new[0])
        new_y = (q_near[1],q_new[1])

        plt.plot(new_x,new_y, color = "blue", linewidth=3)
        plt.title(str(K+1)+" Iterations")
        plt.xlim((0,100))
        plt.ylim((0,100))
        plt.xlabel("X Range")
        plt.ylabel("Y Range")
        '''
        #obstacle generation
        a=np.random.rand()*D
        b=np.random.rand()*D
        area = pow((30 * np.random.rand(5)),2 ) # 0 to 15 point radii
        plt.scatter(a,b, s=area, c ='#000000', alpha=0.5)
        #checking if passing over the lines
        x1=q_rand[0]
        x2=q_new[0]
        y1=q_rand[1]
        y2=q_new[1]
        list(bresenham(x1, y1, x2, y2))
        #getting pixels from coordinates
        for i in range (index): #=255

'''

        plt.pause(0.001)

    plt.show()


if __name__=="__main__":
    main()
    