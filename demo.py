import shaded as sd
from sounddevice import play
import soundfile as sf
import numpy as np
song = sd.space(13)
overall_env = sd.smoothstep(0,1,song) - sd.smoothstep(12,14.8,song)

bass_line = sd.arp(song, [110, 130.81, 146.83, 110]*5) #this time, the "*" means we're repeating
bass_line_wave = sd.smooth_normalize(sd.sin(song, bass_line) * sd.pulse(song, 3, 0.333, normalize=False))

drums = sd.noise(song) * sd.sigmoid(song, 10) * sd.sigmoid(song, 12, 0.25*2*np.pi) * 0.2

melody_a = sd.arp(song, [440, 880, 440, 220])
melody_b = sd.arp(song, [523.25, 392.00, 659.25]*6)
string = sd.saw(song, melody_a) * 0.25 + sd.saw(song, melody_b) * 0.25

full_song = bass_line_wave * sd.smoothstep(2, 3, song) + drums * 0.25 + string * 0.5 * (sd.smoothstep(4, 6, song) - sd.smoothstep(8, 10, song))

sf.write('working/demo.wav',full_song, int(44100), 'FLOAT')