import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import shaded as sp
from shaded.const import SAMPLE_RATE

def play(sound, sample_rate=SAMPLE_RATE):
  """Helper to play serially"""
  sd.play(sound)
  time.sleep(len(sound) / sample_rate)

def save(path, sound, sample_rate=SAMPLE_RATE):
  sf.write(path, sound, int(sample_rate), 'FLOAT')

space = sp.space(3)
long_space = sp.reverb(sp.space(30), -0.25, 8, 0.77)
total_space = sp.space(3*60)


def harmonic(space, base, div=2):
  s = sp.saw(space, base)
  for i in range(1,5*div):
    n = i/div
    s -= sp.saw(space, base*n) * 1./n - sp.saw(space, base/n) * 1./n
  return sp.smooth_normalize(s)

def buzz(space, f):
  return harmonic(space, f+sp.square(space, 3)*sp.sin(space, 0.4), 3)*(sp.sigmoid(space, 0.5, np.pi) - sp.sin(space, 0.25))*sp.square(space,0.333)


def arp_buzz(space):
  return sp.reverb(buzz(space, sp.arp(space, [220, 440, 220, 440, 225, 440, 218])) * 0.2 +
    buzz(space, sp.arp(space, [60, 65, 68, 64])) * sp.sin(space, 0.1) * 0.2, 0.125, 5)

def shaped_noise(space):
  return (
    np.fmod(sp.noise(space), sp.sin(space, 60)) * 0.2 * sp.sigmoid(space, sp.sin(space, 0.45) * 0.1)
      * sp.square(space, 1)
  )

def tone(space, seq):
  return (sp.square(space, seq, shift=sp.sin(space, 0.1)*np.pi) + sp.square(space, seq+2)/2.)*np.clip(sp.square(space, 0.25), 0., 1.)*sp.sin(space, 0.25)

#save('working/rippletone.wav', tone(long_space, sp.arp(long_space, [115, 122, 104,99]*3)))

print("Starting")
play(
  (sp.gate(space, 5) * (shaped_noise(long_space) + arp_buzz(long_space)) +
    tone(long_space, sp.arp(long_space, [220,110,220,440]*9))) * 0.125
)
print("Done")