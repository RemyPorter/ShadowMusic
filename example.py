import time
import shadow as sp
import sounddevice as sd

def play(sound, sample_rate=sp.SAMPLE_RATE):
  """Helper to play serially"""
  sd.play(sound)
  time.sleep(len(sound) / sample_rate)

space = sp.space(3)
play(sp.sin(space, 440)) #concert A sin wave for three seconds
play(sp.sin(space, 440) * (sp.sin(space, 1.) / 2. + 0.5)) #beating w/ a sin wave
play(sp.sigmoid(sp.sin(space,60), sp.sin(space,60)+220)*0.5) #buzzing by modulating frequency
play(sp.sigmoid(sp.sin(space,60), sp.sin(space,60)+220)*(sp.sin(space, 2.) / 2. + 0.5)*0.5) #buzzing by modulating frequency, with a beat
play(sp.noise(space)*(sp.sigmoid(space, 2) - sp.sin(space,0.4) - sp.sin(space,0.25,1)) + sp.sin(space, 440)*(sp.sin(space,3)-sp.sin(space,4.25)) + sp.sin(space,310)*(sp.sin(space,3,1.254))) #complex beat