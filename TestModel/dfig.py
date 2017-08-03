import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib import animation 
# First set up the figure, the axis, and the plot element we want to animate 
fig = plt.figure() 
ax = plt.axes(xlim=(0, 10), ylim=(0, 20))
line, = ax.plot([], [], lw=2) 
# initialization function: plot the background of each frame 
def init(): 
  line.set_data([], []) 
  return line, 
# animation function. This is called sequentially 
# note: i is framenumber
value=range(0,20)
def animate(i):
  print i
  end=i+10
  x = range(0,10)
  y = value[i:end]
  print x
  print y
  line.set_data(x, y) 
  return line, 
# call the animator. blit=True means only re-draw the parts that have changed. 
anim = animation.FuncAnimation(fig, animate, init_func=init, 
                frames=10, interval=1000, blit=True)
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264']) 
plt.show() 