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

def bass_prog(space, reps=3):
  prog = sp.arp(space, [60, 65, 80, 85, 60, 65, 80, 83, 70, 72]*reps)
  return sp.pulse(space, prog, 0.45)

def bass_beat(space):
  return sp.gated_pulse(space, 2, 0.8) + sp.gated_pulse(space, 5, 0.25)

def bass_enter_exit(space, reps=3):
  return sp.arp(space, [1, 1, 1, 1.25, 0.75, 0.25, 0.25, 0.125, 0.125, 0, 0, 0, 0.5, 0.5, 0.5, 0.25, 0.25, 1, 1, 1]*reps)

def drum_line(space):
  return (sp.saw(space+sp.noise(space)*0.004+sp.saw(space,12)*5., 40) *
     sp.gated_pulse(space, 4, 0.25) * sp.sin(space, 4) + sp.gated_pulse(space, 4, 0.125, np.pi*0.1))

def melody_gate(space, start, stop):
  return np.fmin(sp.gate(space, start), np.logical_not(sp.gate(space, stop))) * sp.gated_pulse(space, 1./5., 0.8)

def melody_pattern(space, base=220, low=0.2, high=4., steps=5, reps=3):
  mults = np.linspace(low, high, steps) * base
  np.random.shuffle(mults)
  return sp.arp(space, np.repeat(mults, reps))

def melody_rhthym(space):
  return sp.sigmoid(space, 0.25) * sp.gated_pulse(space, 1, 0.3333) + sp.sin(space, 1, np.pi)

def melody(space):
  return sp.scale_normalize(
    sp.reverb(sp.saw(song, melody_pattern(song, steps=7, reps=25)), 1./60., 5) -
    sp.reverb(sp.saw(song, melody_pattern(song, 0.5, 2, 15, 7)), 1./55., 5)
  ) * sp.square(space+sp.sin(space, 1./40.)*40, 40)

song = sp.space(120)
bass_line = sp.reverb(bass_prog(song, 20) * bass_beat(song) * bass_enter_exit(song, 7), 1./15., 10)
drums = sp.reverb(drum_line(song), -1./30., 7)
mel = melody(song) * melody_rhthym(song) * melody_gate(song, 3., 100)
play(
  (bass_line + drums * 0.5 + mel) * sp.sin(song, 1./60.)
)
