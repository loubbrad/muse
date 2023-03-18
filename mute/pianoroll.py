import mido
import midi

# TODO:
# - Clean up code (quotations)
# - Add more comments and doublecheck docstrings


class PianoRoll():
    """Container for piano-roll objects, includes data and meta-data.

    Args:
        p_roll (list[list]): Piano-roll indicating when notes are on.
        meta_data (dict): _description_
    """

    def __init__(self, roll: list[list], meta_data: list):
        """Initialises PianoRoll with data and metadata."""

        self.roll = roll
        self.meta_data = meta_data

    def to_midi(self):
        """Inplace version of pianoroll.to_midi().

        Returns:
            mido.MidiFile: MidiFile parsed from self.
        """
        return to_midi(self)


def to_midi(piano_roll: PianoRoll):
    """Parses a PianoRoll object into a mid.MidiFile object.

    Args:
        piano_roll (PianoRoll): piano-roll to be parsed:

    Returns:
        mido.MidiFile: parsed MidiFile.
    """

    def _turn_on(track: mido.MidiTrack, notes: list):
        """Adds all notes as note_on events to track."""
        for note in notes:
            track.append(mido.Message('note_on', channel=0,
                         note=note, velocity=100, time=0))
            on_notes.append(note)

    def _turn_off(track: mido.MidiTrack, notes: list):
        """Adds notes as note_off events to track."""
        for note in notes:
            track.append(mido.Message('note_off', channel=0,
                         note=note, velocity=100, time=0))
            on_notes.remove(note)

    ticks_per_step = piano_roll.meta_data['ticks_per_step']
    div = piano_roll.meta_data['div']

    mid = mido.MidiFile(type=1)
    mid.ticks_per_beat = div * ticks_per_step

    meta_track = mido.MidiTrack()
    track = mido.MidiTrack()
    mid.tracks.append(meta_track)
    mid.tracks.append(track)

    # Add meta events to meta_track
    meta_track.append(mido.Message("program_change", program=0, time=0))
    piano_roll.meta_data['meta_events'].sort(key=lambda v: v['time'])
    prev_time = 0
    for meta_event in piano_roll.meta_data['meta_events']:
        meta_track.append(
            mido.MetaMessage(type=meta_event["type"],
                             time=(meta_event["time"] - prev_time)*(
                                 ticks_per_step),
                             **meta_event["data"])
        )
        prev_time = meta_event['time']

    # Add note events to track
    delta_t = 0
    on_notes = []
    for curr_notes in piano_roll.roll:
        turn_on_notes = [note for note in curr_notes if note not in on_notes]
        turn_off_notes = [note for note in on_notes if note not in curr_notes]

        if turn_off_notes == [] and turn_on_notes == []:
            delta_t += ticks_per_step
        else:
            ind = len(track)
            _turn_off(track, turn_off_notes)
            _turn_on(track, turn_on_notes)
            track[ind].time += delta_t
            delta_t = ticks_per_step

    track.append(mido.MetaMessage('end_of_track', time=0))

    return mid


def test():
    mid = mido.MidiFile('mute/pathetique-1.mid')
    p_roll = midi.to_pianoroll(mid, 8)
    new_mid = p_roll.to_midi()
    new_mid.save('test.mid')


if __name__ == "__main__":
    test()
