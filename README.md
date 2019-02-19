# Shaded Music
A time-dimension-based synthesizer built using NumPy, for Python3.

## Shaded Philosophy
Shaded is designed to generate audio by broadcasting functions across a temporal space. This means audio generation happens entirely in memory, and instead of using musical concepts like "beats" we tend to think in terms of intersecting and overlapping waveforms. It is, in its own way, designed to emulate the behavior of OpenGL fragment shaders, but for audio.

Yes, that's a difficult and weird way to think about generating audio, but what it demonstrates is that a surpisingly small amount of code can generate a wide variety of complex sounds. This is less a tool for musical composition and more a tool for acoustic exploration.

# Getting Started
## Space and Time
The entry point for `shaded` is `space`. The `space` function generates a linear series in the time dimension, from `0.` to `n` where `n` is the `duration * sample rate`. So, a 3 second space, at 44100Hz sample rate, will be an array 132,300 elements long, and each element will store its offset in seconds. A `space` is the input for most operations.

**Example**:
```python
>>> import shaded as sp
>>> sp.space(3)
array([0.00000000e+00, 2.26757370e-05, 4.53514739e-05, ...,
       2.99993197e+00, 2.99995465e+00, 2.99997732e+00])
```

## Sound
Once you have a `space`, you can use that space to generate audio. For example, to convert a space into a 440hz sine wave, you can simply use the `sin` function.

```python
>>> import shaded as sp
>>> space = sp.space(3) # a 3 second linear space in the time dimension
>>> sp.sin(space, 440) # returns a 3 second sinusoidal wave generated from that space
array([ 0.        ,  0.06264832,  0.12505052, ..., -0.18696144,
       -0.12505052, -0.06264832])
>>> import sounddevice as sd
>>> sd.play(sp.sin(space, 440)) #play the wave using the sounddevice library
```

### Generation Functions
* `sin(space,freq,shift=0)`: apply a sin function to the contents of `space`, scaling so that, if space is timelike, the resulting wave is `freq`
* `square(space,freq,shift=0)`: as sin, but a square wave
* `sigmoid(space,freq,shift=0)`: as sin, but a sigmoid wave
* `noise(space)` will generate random noise of the same size as the space, ignoring the contents of the space

### Effects
* `delay(space,duration,wet=0.5,dry=0.5)`: copies-and-rolls the input space, then mixes it with the original
* **gain**: just multiply a space by a value, e.g. `sp.sin(space,440) * 3.`.
* **clip**: use `np.clip`, e.g.: `np.clip(sp.sin(space,440)*5.,-1., 1.)`
* **reverse**: use slices, e.g.: `space[::-1]`


## Complex Outcomes
With a few basic waveforms and noise generation functions, we can create surprisingly complex sounds. For example, something similar to a beat could be created by multiplying waves together:

```python
sd.play(sp.noise(space)*sp.sigmoid(space, 2)*0.5) #the 0.5 at the end controls the overall amplitude of the wave
```

By adding and subtracting waves together, we can create very complex sounds:

```python
sd.play(sp.noise(space)*(sp.sigmoid(space, 2) - sp.sin(space,0.4) - sp.sin(space,0.25,1)) + sp.sin(space, 440)*(sp.sin(space,3)-sp.sin(space,4.25)) + sp.sin(space,310)*(sp.sin(space,3,1.254))) #complex beat
```

Multiplication and division, of course, also work:

```python
sd.play((sp.sin(space, 17) / sp.sin(space, 3)) * 0.2 * sp.sin(space, 440))
```

Also, because of numpy's broadcasting, you can pass waveforms in the frequency parameter. This creates an odd wobble sound:

```python
sd.play(sp.sin(space, sp.sin(space, 2) * 220))
```

Or, go really weird, and combine broadcasting with non-temporal spaces as inputs, like so:

```python
sd.play(sp.sin(sp.sin(space, 2) * 2 * np.pi, sp.sin(space, 20) * 3))
```

# Simplicity Itself
Take a look at the code in `shaded/__init.py__`. It's *extremely* simple code. It's so simple, in fact, that it's barely worth even writing unit tests (though as this project matures, I'll certainly add some actual testing). By using NumPy's broadcastable functions, `mgrid` and the like, it's *extremely* easy to create audio, and to process that audio in interesting ways.