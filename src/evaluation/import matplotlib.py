import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

frames = [np.random.rand(84, 84, 3) for _ in range(30)]  # Dummy changing frames

fig = plt.figure()
plt.axis("off")
ims = [[plt.imshow(f, animated=True)] for f in frames]
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
ani.save("test.gif", writer="pillow")
