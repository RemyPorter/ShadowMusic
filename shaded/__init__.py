import numpy as np
from .const import SAMPLE_RATE

def smoothstep(edge0, edge1, x):
  """
  Hermite Interpolation, lifted from OpenGL shaders.

  Converts a data stream into a sigmoid from edge0 to edge1 (outputting in the range [0.,1.])
  """
  t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)

def step(space):
  return np.heaviside(space, 0)

def space(duration, sample_rate=SAMPLE_RATE):
  """Generate a temporal space of `duration` seconds at `sample_rate` sampling frequency"""
  return np.mgrid[0:int(sample_rate*duration)] / sample_rate

def noise(space):
  """Output random noise in the same range as our temporal space"""
  return (np.random.random(len(space)) - 0.5) * 2.

def sin(space, freq,shift=0):
  """Convert an input space to a sinusoidal wave with `freq` frequency. `shift` moves the phase."""
  return np.sin(2*np.pi*space*freq+shift)

def sigmoid(space, freq, shift=0):
  """Convert an input space to a sigmoidal wave with `freq` frequency. `shift` moves the phase."""
  return smoothstep(0, 1., sin(space, freq, shift)) + smoothstep(-1., 0., sin(space, freq, shift))

def square(space, freq, shift=0):
  """Simple 50% duty cycle square wave"""
  s = sin(space, freq, shift)
  return 2.*step(s)-1.

def saw(space, freq, shift=0):
  """Sawtooth wave"""
  return np.arctan(
    1. / np.tan(
      space * np.pi * freq + shift
    )
  )
  
def delay(space, distance, wet=0.5, dry=0.5, sample_rate=SAMPLE_RATE):
  """Delay and mix"""
  d = int(distance * sample_rate)
  return np.roll(space, d) * wet + space * dry

def reverb(space, delay_time, iterations, falloff=0.5, sample_rate=SAMPLE_RATE):
  """
  Simple reverb with a gradual decay over time. `delay_time` is the echo time, `iterations` is how many echos to 
  generate, `falloff` is how much quieter each echo is
  """
  res = np.zeros(space.shape)
  d = np.copy(space)
  for i in range(iterations):
    d = delay(d, delay_time, wet=falloff, dry=0., sample_rate=SAMPLE_RATE)
    res += d
  return res

def arp(space, sequence):
  """
  Breaks an arbitrary space up into equal sized units of sequence.
  """
  res = np.copy(space) #you need a copy because array_split changes the array in place
  split = np.array_split(res, len(sequence)) #
  for v,s in zip(sequence,split):
    s[::] = v
  return res

def repeat(space, data_space):
  """Repeat a dataset to fill our entire space via tiling"""
  return np.tile(data_space, int(len(space)/len(data_space)))
