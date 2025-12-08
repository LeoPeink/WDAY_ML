from matplotlib import pyplot as plt


def onclick(event):
   print('The Click Event Triggered!')
   print('You clicked on: x=%d, y=%d' % (event.x, event.y))

fig, ax = plt.subplots(figsize=(7, 4))
ax.text(0.1, 0.5, 'Click me anywhere on this plot!', dict(size=20))
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()