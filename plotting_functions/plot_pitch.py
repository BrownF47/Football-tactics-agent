import matplotlib.pyplot as plt
import numpy as np

def plot_pitch():

    pitch_length = 100
    pitch_width = 70

    # boarder #
    plt.plot([0,pitch_length], [0,0], color='black')
    plt.plot([0,pitch_length], [pitch_width,pitch_width], color='black')
    plt.plot([0,0], [0,pitch_width], color='black')
    plt.plot([pitch_length,pitch_length], [0,pitch_width], color='black')

    # halfwayline # 
    plt.plot([0.5*pitch_length, 0.5*pitch_length], [0,pitch_width], color='black')

    # centre circle #
    r = 9.15
    theta = np.linspace(0, 2*np.pi, 200, True)  
    plt.plot(r*np.cos(theta) + 0.5*pitch_length, r*np.sin(theta) + 0.5*pitch_width, color='black')

    # goals #
    goal_size = 7.32
    goal_depth = 2
    # left #
    plt.plot([-goal_depth,-goal_depth],[0.5*pitch_width - 0.5*goal_size, 0.5*pitch_width + 0.5*goal_size], color='black')
    plt.plot([0, -goal_depth], [0.5*pitch_width - 0.5*goal_size,0.5*pitch_width - 0.5*goal_size], color='black')
    plt.plot([0, -goal_depth], [0.5*pitch_width + 0.5*goal_size,0.5*pitch_width + 0.5*goal_size], color='black')
    # right #
    plt.plot([pitch_length+goal_depth,pitch_length+goal_depth],[0.5*pitch_width - 0.5*goal_size, 0.5*pitch_width + 0.5*goal_size], color='black')
    plt.plot([pitch_length, pitch_length+goal_depth], [0.5*pitch_width - 0.5*goal_size,0.5*pitch_width - 0.5*goal_size], color='black')
    plt.plot([pitch_length, pitch_length+goal_depth], [0.5*pitch_width + 0.5*goal_size,0.5*pitch_width + 0.5*goal_size], color='black')

    # 18 yrd box # 
    box_width = 40.32
    box_length = 16.5
    # left #
    plt.plot([box_length, box_length], [0.5*pitch_width - 0.5*box_width,0.5*pitch_width + 0.5*box_width], color='black')
    plt.plot([0,box_length],[0.5*pitch_width - 0.5*box_width, 0.5*pitch_width - 0.5*box_width], color='black')
    plt.plot([0,box_length],[0.5*pitch_width + 0.5*box_width, 0.5*pitch_width + 0.5*box_width], color='black')
    # right # 
    plt.plot([pitch_length-box_length,pitch_length-box_length], [0.5*pitch_width - 0.5*box_width,0.5*pitch_width + 0.5*box_width], color='black')
    plt.plot([pitch_length,pitch_length-box_length],[0.5*pitch_width - 0.5*box_width, 0.5*pitch_width - 0.5*box_width], color='black')
    plt.plot([pitch_length,pitch_length-box_length],[0.5*pitch_width + 0.5*box_width, 0.5*pitch_width + 0.5*box_width], color='black')

    # 6 yrd box # 
    box_width = 18.32
    box_length = 5.5
    # left #
    plt.plot([box_length, box_length], [0.5*pitch_width - 0.5*box_width,0.5*pitch_width + 0.5*box_width], color='black')
    plt.plot([0,box_length],[0.5*pitch_width - 0.5*box_width, 0.5*pitch_width - 0.5*box_width], color='black')
    plt.plot([0,box_length],[0.5*pitch_width + 0.5*box_width, 0.5*pitch_width + 0.5*box_width], color='black')
    # right # 
    plt.plot([pitch_length-box_length,pitch_length-box_length], [0.5*pitch_width - 0.5*box_width,0.5*pitch_width + 0.5*box_width], color='black')
    plt.plot([pitch_length,pitch_length-box_length],[0.5*pitch_width - 0.5*box_width, 0.5*pitch_width - 0.5*box_width], color='black')
    plt.plot([pitch_length,pitch_length-box_length],[0.5*pitch_width + 0.5*box_width, 0.5*pitch_width + 0.5*box_width], color='black')

    # penalty spots # 
    plt.scatter([11], [0.5*pitch_width], color='black', s=5)
    plt.scatter([pitch_length-11], [0.5*pitch_width], color='black', s=5)



    # set aspect ratio equal # 
    plt.gca().set_aspect('equal')

def plot_half_pitch():

    pitch_length = 50
    pitch_width = 70

    # boarder #
    plt.plot([0,pitch_length], [0,0], color='black')
    plt.plot([0,pitch_length], [pitch_width,pitch_width], color='black')
    plt.plot([0,0], [0,pitch_width], color='black')
    plt.plot([pitch_length,pitch_length], [0,pitch_width], color='black')

    # goal #
    goal_size = 7.32
    goal_depth = 2
    
    plt.plot([pitch_length+goal_depth,pitch_length+goal_depth],[0.5*pitch_width - 0.5*goal_size, 0.5*pitch_width + 0.5*goal_size], color='black')
    plt.plot([pitch_length, pitch_length+goal_depth], [0.5*pitch_width - 0.5*goal_size,0.5*pitch_width - 0.5*goal_size], color='black')
    plt.plot([pitch_length, pitch_length+goal_depth], [0.5*pitch_width + 0.5*goal_size,0.5*pitch_width + 0.5*goal_size], color='black')

    # centre circle #
    r = 9.15
    theta = np.linspace(-np.pi/2, np.pi/2, 100, True)  
    plt.plot(r*np.cos(theta), r*np.sin(theta) + 0.5*pitch_width, color='black')

    # 18 yrd box # 
    box_width = 40.32
    box_length = 16.5
    
    plt.plot([pitch_length-box_length,pitch_length-box_length], [0.5*pitch_width - 0.5*box_width,0.5*pitch_width + 0.5*box_width], color='black')
    plt.plot([pitch_length,pitch_length-box_length],[0.5*pitch_width - 0.5*box_width, 0.5*pitch_width - 0.5*box_width], color='black')
    plt.plot([pitch_length,pitch_length-box_length],[0.5*pitch_width + 0.5*box_width, 0.5*pitch_width + 0.5*box_width], color='black')

    # 6 yrd box # 
    box_width = 18.32
    box_length = 5.5

    plt.plot([pitch_length-box_length,pitch_length-box_length], [0.5*pitch_width - 0.5*box_width,0.5*pitch_width + 0.5*box_width], color='black')
    plt.plot([pitch_length,pitch_length-box_length],[0.5*pitch_width - 0.5*box_width, 0.5*pitch_width - 0.5*box_width], color='black')
    plt.plot([pitch_length,pitch_length-box_length],[0.5*pitch_width + 0.5*box_width, 0.5*pitch_width + 0.5*box_width], color='black')

    # penalty spots # 
    plt.scatter([pitch_length-11], [0.5*pitch_width], color='black', s=5)    

    # set aspect ratio equal # 
    plt.gca().set_aspect('equal')



if __name__ == "__main__":
    plot_pitch()
    plt.show()
    plot_half_pitch()
    plt.show()