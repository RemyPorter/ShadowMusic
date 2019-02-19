import time
import numpy as np
import shaded as sp
import sounddevice as sd
import soundfile as sf

def play(sound, sample_rate=sp.SAMPLE_RATE):
  """Helper to play serially"""
  sd.play(sound)
  time.sleep(len(sound) / sample_rate)

space = sp.space(3)

one_hz = (sp.sin(space, 1.) / 2. + 0.5) #beating
concertA = sp.sin(space, 440) #concert A sin wave for three seconds
modulatedA = concertA * one_hz #beating w/ a sin wave
buzz = sp.sigmoid(sp.sin(space,60), sp.sin(space,60)+220)*0.5 #buzzing by modulating frequency
div = (sp.sin(space, 17) / sp.sin(space, 3)) * 0.2 * concertA #dividing signals to generate complex waveforms
complex_beat = sp.noise(space)*(sp.sigmoid(space, 2) - sp.sin(space,0.4) - sp.sin(space,0.25,1)) + sp.sin(space, 440)*(sp.sin(space,3)-sp.sin(space,4.25)) + sp.sin(space,310)*(sp.sin(space,3,1.254)) #complex beat
fmod = sp.sin(space, sp.sin(space, 2) * 220) #because of broadcasting, you can use waves to control the frequency of other waves
deriv = sp.sin(sp.sin(space, 2) * 2 * np.pi, sp.sin(space, 20) * 3) #the sin of a sin
square = sp.square(space,440) #square wave
modulated_square = sp.square(space,440)*sp.sin(space,440)-sp.square(space,443,0.5) #combining waves
beating = (sp.sin(space,440) + sp.sin(space,110)) * sp.square(space,90000) * sp.square(space,1.50) * sp.sin(space,3) * sp.square(space,11) * sp.sin(space,0.5)

play(concertA) 
play(modulatedA)
play(square)
play(modulated_square)
play(buzz) 
play(buzz*one_hz) 
play(div)
play(complex_beat) 
play(fmod) 
play(deriv) #this is just nutty
play(complex_beat[::-1])
play(beating)


# Uncomment this block to save the data to files
def save(name, data):
  sf.write(name, data, sp.SAMPLE_RATE, 'FLOAT')
"""
save('samples/concertA.wav', concertA)
save('samples/modulatedA.wav', modulatedA)
save('samples/square.wav', square)
save('samples/modulated_square.wav', modulated_square)
save('samples/buzz.wav', buzz)
save('samples/modulatedBuzz.wav', buzz*one_hz)
save('samples/div.wav', div)
save('samples/complex.wav', complex_beat)
save('samples/fmod.wav', fmod)
save('samples/deriv.wav', deriv)
save('samples/beating.wav', beating)
"""
