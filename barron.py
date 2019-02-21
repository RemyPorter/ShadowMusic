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

space = sp.space(3)
long_space = sp.space(30)

pitcher = sp.saw(space, 1., np.pi/2.)*5.

rising_saw_30s = np.tile(sp.reverb(
    sp.sigmoid(space, 220+pitcher) + sp.sigmoid(space, 442-pitcher) + sp.sigmoid(space, 106-pitcher),
    0.25, 15
  ), 10)
rising_saw_30s *= sp.sin(long_space, 1./15.) * sp.sin(long_space, 1./17., shift=np.pi/4.)

seq = np.array([125, 225, 325, 175, 90, 95]*40)
np.random.shuffle(seq)
melody = sp.saw(long_space, sp.arp(long_space,seq)) * np.clip(sp.square(long_space,0.425), 0., 1.) * sp.sin(long_space,0.444) * sp.sin(long_space,0.01) * 0.45
#play(
sf.write('working/barron.wav',
  sp.reverb(rising_saw_30s - melody, 0.25, 8, 0.45),
  int(SAMPLE_RATE), 'FLOAT')
#)