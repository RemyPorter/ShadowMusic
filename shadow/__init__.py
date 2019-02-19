import numpy as np

SAMPLE_RATE=44100

def smoothstep(edge0, edge1, x):
  t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)

def space(duration, sample_rate=SAMPLE_RATE):
  return np.mgrid[0:int(sample_rate*duration)] / sample_rate

def noise(space):
  return (np.random.random(len(space)) - 0.5) * 2.

def sin(space, freq,shift=0):
  return np.sin(2*np.pi*space*freq+shift)

def sigmoid(space, freq, shift=0):
  return smoothstep(0, 1., sin(space, freq, shift)) + smoothstep(-1., 0., sin(space, freq, shift))
  
