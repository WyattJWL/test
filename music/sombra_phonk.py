import numpy as np, wave, subprocess, os
GAIN = dict(zip(("drums", "perc", "bass", "bell", "pad", "fx"), map(float, os.environ.get("GAINS", "0.65 4.0 0.42 1.9 1.0 0.5").split())))
from scipy.signal import butter, sosfilt, fftconvolve

sr = 44100; bpm = 82.5; beat = 60 / bpm; s16 = beat / 4; bar = 4 * beat
BARS = 25; N = int(sr * bar * BARS) + sr * 3
rng = np.random.default_rng(11)

# separate stems so we can filter / cut each one
stems = {k: np.zeros((N, 2)) for k in ("drums", "perc", "bass", "bell", "pad", "fx")}
def add(stem, s, t0, pan=.5, g=1.):
    i = int(t0 * sr)
    if i >= N or i < 0: return
    s = s[:N - i]
    stems[stem][i:i + len(s), 0] += s * g * (1 - pan) * 1.4
    stems[stem][i:i + len(s), 1] += s * g * pan * 1.4

def hz(m): return 440 * 2 ** ((m - 69) / 12)
def filt(x, kind, fc, order=2):
    return sosfilt(butter(order, fc, kind, fs=sr, output="sos"), x, axis=0)
def noise(n): return rng.standard_normal(n)

# ---------- instruments ----------
def kick():
    n = int(.32 * sr); t = np.arange(n) / sr
    f = 48 + 150 * np.exp(-t * 40); ph = 2 * np.pi * np.cumsum(f) / sr
    click = filt(noise(n), "highpass", 3000) * np.exp(-t * 300) * .3
    return np.tanh(3 * np.sin(ph) * np.exp(-t * 10)) + click
def clap():
    n = int(.35 * sr); t = np.arange(n) / sr
    e = sum(np.exp(-np.clip(t - d, 0, None) * 80) * (t >= d) for d in (0, .01, .021)) * .5 + np.exp(-t * 14) * .45
    return filt(noise(n), "bandpass", [900, 6000]) * e
def hat(open_=False):
    n = int((.22 if open_ else .045) * sr); t = np.arange(n) / sr
    return filt(noise(n), "highpass", 7000) * np.exp(-t * (13 if open_ else 110)) * .5
def cowbell(m, dur=.4):
    n = int(dur * sr); t = np.arange(n) / sr; f = hz(m)
    y = np.sign(np.sin(2 * np.pi * f * t)) * .6 + np.sign(np.sin(2 * np.pi * f * 1.48 * t)) * .5
    y = filt(y, "bandpass", [f * .9, 7000])
    return np.tanh(y * 1.6) * np.exp(-t * 9) * np.minimum(t / .001, 1) * .6
def sub808(m, dur, glide_from=None):
    n = int(dur * sr); t = np.arange(n) / sr
    f = np.full(n, hz(m))
    if glide_from is not None:
        f = hz(glide_from) + (hz(m) - hz(glide_from)) * np.minimum(t / .12, 1)
    f = f * (1 + .4 * np.exp(-t * 45)); ph = 2 * np.pi * np.cumsum(f) / sr
    y = np.tanh(5 * np.sin(ph)) * .55 + np.sin(ph) * .5
    return y * np.minimum(t / .003, 1) * np.minimum(1, (dur - t) / .04) * np.exp(-t / (dur * 1.2))
def choir(ms, dur):  # dark "ooh" pad
    n = int(dur * sr); t = np.arange(n) / sr; y = np.zeros(n)
    for m in ms:
        for dt in (-.08, 0, .08):
            f = hz(m) * 2 ** (dt / 12) * (1 + .003 * np.sin(2 * np.pi * 5 * t + m))
            ph = 2 * np.pi * np.cumsum(f) / sr
            y += sum(np.sin(k * ph) / k ** 1.3 for k in range(1, 9))
    y = filt(y, "bandpass", [250, 1400])
    return y / (np.abs(y).max() + 1e-9) * np.minimum(t / .8, 1) * np.minimum((dur - t) / .8, 1) * .5
def vox(m, dur=.3):  # chopped vocal-like "ey!"
    n = int(dur * sr); t = np.arange(n) / sr
    f = hz(m) * (1 + .3 * np.exp(-t * 30)); ph = 2 * np.pi * np.cumsum(f) / sr
    y = sum(np.sin(k * ph) / k for k in range(1, 30))
    y = filt(y, "bandpass", [500, 900]) + filt(y, "bandpass", [1800, 2600]) * .6
    return y / (np.abs(y).max() + 1e-9) * np.minimum(t / .005, 1) * np.exp(-t * 9) * .7
def impact():
    n = int(2.5 * sr); t = np.arange(n) / sr
    boom = np.sin(2 * np.pi * np.cumsum(35 + 60 * np.exp(-t * 6)) / sr) * np.exp(-t * 2.2)
    return np.tanh(2 * boom) + filt(noise(n), "lowpass", 3000) * np.exp(-t * 3) * .25
def riser(d):
    n = int(d * sr); t = np.arange(n) / sr
    return filt(noise(n), "bandpass", [2000, 9000]) * (t / d) ** 2.5 * .5

# ---------- harmony / patterns (G harmonic minor) ----------
G, Eb, D, Fs, C, Ab = 0, 3, 2, -1, -7, 1          # semitone offsets from G
bass_prog = [43, 43, 39, 38]                      # G2? -> used an octave down: G1 G1 Eb1 D1
bass_prog = [m - 12 for m in bass_prog]
choir_prog = [[55, 58, 62], [55, 58, 62], [51, 55, 58], [50, 54, 57]]
# my own 2-bar cowbell riff (16th positions over 32 steps, semitones from G5)
riff_a = [(0, G), (3, G), (6, Eb), (8, D), (11, Eb), (14, D), (16, G), (19, G), (22, Fs), (24, D), (27, Eb), (30, D)]
riff_b = [(0, G + 12), (3, G), (6, Eb), (8, Fs), (11, G), (14, Ab), (16, G), (19, Eb), (22, D), (24, C + 12), (27, D), (30, Fs)]
# funk 3+3+2 (in 8ths at double time) spread over 2 bars of 16ths
kick_pos = [0, 6, 12, 16, 22, 28]
bass_pos = [(0, 6), (6, 6), (12, 4), (16, 6), (22, 6), (28, 4)]

def section(b):
    if b < 1: return "intro"
    if b < 9: return "dropA"
    if b < 13: return "break"
    if b < 21: return "dropB"
    return "final"

for b in range(BARS):
    sec = section(b); t0 = b * bar; ch = (b // 2) % 4 if sec != "intro" else 0
    half = b % 2                         # which half of the 2-bar pattern
    riff = riff_b if (sec == "dropB" and (b // 2) % 2) or sec == "final" else riff_a
    # pad
    if sec in ("break", "intro") or b % 2 == 0:
        add("pad", choir(choir_prog[ch], 2 * bar if sec != "break" else bar + .5), t0, .5, .55 if sec == "break" else .35)
    # cowbell riff
    for p, d in riff:
        if half * 16 <= p < half * 16 + 16:
            tt = t0 + (p - half * 16) * s16
            if sec == "break" and p % 6: continue
            add("bell", cowbell(79 + d), tt, .38, .55)
            add("bell", cowbell(79 + d + 12, .25), tt, .62, .12)
    if sec in ("dropA", "dropB", "final"):
        for p in kick_pos:
            if half * 16 <= p < half * 16 + 16:
                add("drums", kick(), t0 + (p - half * 16) * s16, .5, 1.)
        add("perc", clap(), t0 + 2 * beat, .5, .9)
        if sec != "dropA" and half: add("perc", clap(), t0 + 3 * beat + 2 * s16, .5, .45)
        for k in range(16):
            if sec == "dropA" and k % 2: continue
            v = 1 if k % 4 == 0 else .6
            add("perc", hat(k in (6, 14)), t0 + k * s16, .3 + .4 * ((k * 7) % 3) / 2, v * .75)
        if sec != "dropA" and b % 4 == 3:          # hat roll
            for k in range(8): add("perc", hat(), t0 + 3 * beat + k * s16 / 2, .55, .55)
        for p, ln in bass_pos:
            if half * 16 <= p < half * 16 + 16:
                root = bass_prog[ch]
                gl = root + 1 if p in (12, 28) else None   # semitone slide down into root
                add("bass", sub808(root, ln * s16 * .97, gl), t0 + (p - half * 16) * s16, .5, 1.)
        if sec != "dropA":
            for p, d in ((4, G), (10, Eb), (20, D), (26, Fs)):
                if half * 16 <= p < half * 16 + 16:
                    add("pad", vox(67 + d), t0 + (p - half * 16) * s16, .7, .45)
    if sec == "break":
        add("bass", sub808(bass_prog[ch], bar * .9), t0, .5, .35)

# FX layer: impacts on drops, risers before
for db in (1, 13, 21): add("fx", impact(), db * bar, .5, .7)
for rb in (0, 12, 20): add("fx", riser(bar), rb * bar, .5, .8)

# ---------- mixdown with section processing ----------
def sl(b0, b1): return slice(int(b0 * bar * sr), int(b1 * bar * sr))
# break: drums out already; lowpass bells + pad for a "underwater" feel, sweep open at the end
for k in ("bell", "pad"):
    s = sl(9, 13); stems[k][s] = filt(stems[k][s], "lowpass", 900)
# intro bell filtered too
stems["bell"][sl(0, 1)] = filt(stems["bell"][sl(0, 1)], "lowpass", 1500)

# bell/perc presence: gentle high-shelf-ish boost via parallel highpass
stems["bell"] += filt(stems["bell"], "highpass", 2500) * .6
mix = sum(stems[k] * GAIN[k] for k in stems)

# reverb (convolution with decaying noise IR) on everything but sub
ir_n = int(1.6 * sr); ir = noise(ir_n) * np.exp(-np.arange(ir_n) / sr * 4.5); ir /= np.abs(ir).sum() ** .5 * 12
wet_src = stems["bell"] * GAIN["bell"] * .5 + stems["pad"] * GAIN["pad"] * .6 + stems["perc"] * GAIN["perc"] * .15
wet = np.stack([fftconvolve(wet_src[:, c], ir)[:N] for c in range(2)], 1)
mix = mix + filt(wet, "highpass", 250) * .9

# ---------- EDIT FX: reverse returns, cuts, stutter, tape-stop ----------
def reverse_return(x, drop_bar, beats=2, gap=.15):
    a = int(drop_bar * bar * sr); n = int(beats * beat * sr)
    rev = x[a:a + n][::-1].copy() * (np.linspace(0, 1, n) ** 1.6)[:, None]
    x[a - n:a] = x[a - n:a] * .2 + rev
    g = int(gap * sr); x[a - g:a] = 0
for db in (1, 13, 21): reverse_return(mix, db)
# stutter at the end of bar 8 (into the break) and bar 16
def stutter(x, b, beat_idx=3):
    a = int((b * bar + beat_idx * beat) * sr); sl_ = int(s16 * sr); seg = x[a:a + sl_].copy()
    pos = a
    for ln in (sl_, sl_, sl_ // 2, sl_ // 2, sl_ // 4, sl_ // 4, sl_ // 8, sl_ // 8, sl_ // 8, sl_ // 8):
        x[pos:pos + ln] = seg[:ln] * np.linspace(1, .8, ln)[:, None]; pos += ln
    x[pos:a + int(beat * sr)] *= .0
stutter(mix, 8); stutter(mix, 16)
# tape-stop on bar 20 (beats 1-2), right before the reverse return into the final drop
def tape_stop(x, t_start, dur):
    a = int(t_start * sr); n = int(dur * sr); seg = x[a:a + n].copy()
    idx = np.clip(np.cumsum(np.linspace(1, .03, n)), 0, n - 1).astype(int)
    x[a:a + n] = seg[idx] * np.linspace(1, .3, n)[:, None]
tape_stop(mix, 20 * bar, 2 * beat)

# master: glue + soft clip + limiter-ish
mix = filt(mix, "highpass", 25)
mix /= np.abs(mix).max() / 1.6
mix = np.tanh(mix)
end = int(BARS * bar * sr) + int(1.5 * sr); mix = mix[:end]
f = int(3 * sr); mix[-f:] *= np.linspace(1, 0, f)[:, None] ** 1.5
mix[:int(.02 * sr)] *= np.linspace(0, 1, int(.02 * sr))[:, None]
mix /= np.abs(mix).max() / .95

out = "/home/user/test/music/sombra_phonk.wav"
w = wave.open(out, "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-b:a", "256k",
                "-metadata", "title=Sombra", "-metadata", "genre=Brazilian Phonk",
                "/home/user/test/music/sombra_phonk.mp3"], check=True)
print("duration", len(mix) / sr)
