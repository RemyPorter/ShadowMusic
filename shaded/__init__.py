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
  """A shelf function: all entries in space <= 0 become 0, all greater than 0 become 1."""
  return np.heaviside(space, 0)

def gate(space, point):
  """Open (or with a `np.not`, close) a gate based on the absolute magnitude of values in space"""
  return step(np.abs(space) - point)

def smooth_normalize(space):
  """
  Take an arbitrary range and compact it into the range -1,1,
  by applying a `smoothstep`
  
  input sets smaller than -1,1 are left unchanged
  """
  mi = np.min(space)
  mx = np.max(space)
  if mi >= -1. and mx < 1.:
    return space
  return smoothstep(np.min(space), np.max(space), space) * 2. - 1.

def scale_normalize(space):
  """
  Take an arbitrary range and compact it into the range -1,1
  by scaling it relative to its peaks.

  input sets smaller than the input range will be scaled UP
  """
  mx = np.max(np.abs(space))
  return space / mx

def average(space, window, sample_rate=SAMPLE_RATE):
  """
  Average across a duration, "smoothing out" the signal.
  """
  return np.convolve(space, np.ones(int(window*sample_rate)), mode='same') / (window * sample_rate)

def convolve(space, kernel):
  """Convolve the signal using a 1D kernel"""
  return np.convolve(space, kernel, mode='same')

def space(duration, sample_rate=SAMPLE_RATE):
  """Generate a temporal space of `duration` seconds at `sample_rate` sampling frequency"""
  return np.linspace(0, duration, duration*sample_rate)

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
  """Generate a square wave by gating a sin"""
  return np.sign(sin(space, freq, shift))

def pulse(space, freq, duty_cycle=0.5, shift=0, normalize=True):
  """
  Generate a rectangular wave with a given duty cycle 

  This wave may either be in the range [0,1] (if normalize==False)
  or may be in the range [-1,1] (if normalize==True, the default)
  """
  s = square(space, freq, shift)
  t = square(space, freq, shift+duty_cycle*2*np.pi)
  wav = np.heaviside(s-t,0)
  if normalize:
    return wav * 2. - 1
  return wav

def gated_pulse(space, freq, duty_cycle, shift=0):
  """A convenience wrapper around pulse which returns a non-normalized pulse. Good for gates"""
  return pulse(space, freq, duty_cycle, shift, False)

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
  res = np.zeros(space.shape)
  split = np.array_split(res, len(sequence)) #
  for v,s in zip(sequence,split):
    s[::] = v
  return res

def repeat(space, data_space):
  """Repeat a dataset to fill our entire space via tiling"""
  return np.tile(data_space, int(len(space)/len(data_space)))
