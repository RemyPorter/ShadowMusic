# Build a Synth in 5 Lines (of your own code)
About:
* Remy Porter
* "Creative Coder" @ Iontank
* Amateur Violinist
* Incredibly Lazy

# What We're Going to Do
* Talk about music and sound
* Talk about numbers, music, and NumPy
* Make some bleeps and bloops

# Music
## What Is Music?
* [image of sheet music] (notes, beats, rhythms)
* [audio file example]
* [instrument]

## Bebe and Louis Barron
* A system
* An organism
* A signal

## Time Varying Signal
* [chart of signal]
  
# Time
* A time varying signal must vary by time
* We're going to sample time
* `one_second = np.linspace(0, 1, 44100)`

## Space
```
def space(duration, sample_rate=SAMPLE_RATE):
  """Generate a temporal space of `duration` seconds at `sample_rate` sampling frequency"""
  return np.linspace(0, duration, duration*sample_rate)
```

# Sin
* I know what a sin wave is
* NumPy has a sin method built in

## It Hertz
* One hertz is one cycle per second
* `np.sin(2*np.pi*one_second*440)`

## Broadcasting
* The power of NumPy
* `np.array([0.25, 0.5, 0.75]) * 2 * np.pi`
* So important to everything we're about to do

## More Examples of Broadcasting
* `np.array([0.25, 0.5, 0.75]) + np.array([0.25, 0.5, 0.75])`

## Back to the sin
* [diagrams of time vs. sin]

## sin
```
def sin(space, freq, shift=0)
  """Convert an input space to a sinusoidal wave with `freq` frequency. `shift` moves the phase."""
  return np.sin(2*np.pi*space*freq+shift)
```

# We Just Made Some Noise
```
a = sin(one_second, 440)
c = sin(one_second, 523.35)
e = sin(one_second, 659.25)
chord = a + c + e
```

## Wait, that's ugly
```
chord = (a + c + e) / 3 #broadcasting!
```

# Broadcasting Again
* We can add arrays together
* We can multiply arrays

## Wait, multiply? I have an idea
```
four_seconds = space(4)
a = sin(four_seconds, 440)
hz1 = sin(four_seconds, 1)
beating = a * hz1
```

## Can we make it more complicated?
```
c = sin(four_seconds, 523.35)
e = sin(four_seconds, 659.25)
beated = (a * sin(four_seconds, 0.66666) +
  c * sin(four_seconds, 0.7) +
  e * sin(four_seconds, 2.5)) / 3.
```

# Sin of the times?
* What other waveforms might we want?
* Square? Sawtooth? Triangle?

## Hip to be Square
* Square is just a sin wave push to the EXTREMEs
* It's POSITIVE or it's NEGATIVE and it's NEVER INBETWEEN!

## Sign of the Square Wave
`np.sign([-0.5, 0.5])`

```
def square(space, freq, shift=0):
  """Generate a square wave by gating a sin"""
  return np.sign(sin(space, freq, shift))
```

## Square Waves
```
sqA = square(four_seconds, 440) * 0.5
beated = sqA * hz1
```

## Squares as Beats?
```
np.heaviside([-1, -0.5, 0, 0.5, 1], 0)
```

```
bpm120 = np.heaviside(square(four_seconds, 2), 0)
```

## Pulse
* Can we combine square waves?

```
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
```

## Complex Rhythm


# Notes
## Broadcasting the Broadcasted
```
sin(one_second, 440+sin(one_second, 2)*10)
```

`440+sin(one_second, 2)*10`

## Sequencing
```
buffer = np.zeros(four_seconds.shape)
sample_rate = 44100
buffer[::] = 440 #fill the whole thing
buffer[sample_rate:sample_rate*2] = 523.35 #second second
buffer[sample_rate*2:sample_rate*3] = 659.25
square(four_seconds, buffer)
```

