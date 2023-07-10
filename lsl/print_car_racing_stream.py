"""Example program to demonstrate how to read a multi-channel time-series
from LSL in a chunk-by-chunk manner (which is more efficient)."""

from pylsl import StreamInlet, resolve_streams

stream_name_token = 'Obstacle'

streams = resolve_streams()

print("="*50)
for s in streams:
    print(s.name(), s.type())
    if stream_name_token in s.name():
        inlet = StreamInlet(s)

while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    chunk, timestamps = inlet.pull_chunk()
    if timestamps:
        print(timestamps, chunk)