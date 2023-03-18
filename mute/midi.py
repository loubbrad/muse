import math
import mido
import collections


from pianoroll import PianoRoll

# TODO:
# - Add proper exception handling.


def to_pianoroll(mid: mido.MidiFile, div: int):
    """Parses a mido.MidiFile object into a PianoRoll object.

    Args:
        mid (mido.MidiFile): Midi to be converted to piano-roll.
        div (int): Integer to subdivide each beat by.

    Returns:
        PianoRoll: mid as a PianoRoll object.
    """

    def _get_metadata(mid: mido.MidiFile, ticks_per_step: int, div: int):
        """Returns list of relevant meta-data present in mid.

        Note that it is expected that mid has events with time in absolute
        units.

        Args:
            mid (mido.MidiFile): mid object to be parsed (absolute time).
            ticks_per_step (int): number of midi ticks per piano-roll step.
            div (int): Amount that each beat is divided.

        Returns:
            dict: extracted relevant metadata.
        """
        meta_data = {"ticks_per_step": ticks_per_step}
        meta_data["div"] = div

        meta_events = []
        for track in mid.tracks:
            for event in track:
                if event.type == "set_tempo":
                    meta_event = {}
                    meta_event['type'] = "set_tempo"
                    meta_event['time'] = event.time // ticks_per_step
                    meta_event['data'] = {"tempo": event.tempo}
                elif event.type == "key_signature":
                    meta_event = {}
                    meta_event['type'] = "key_signature"
                    meta_event['time'] = event.time // ticks_per_step
                    meta_event['data'] = {"key": event.key}
                else:
                    continue

                # Check if meta event is unique
                occured = False
                for event in meta_events:
                    if meta_event['type'] == event['type'] and (
                        meta_event['time'] == event['time'] and (
                            meta_event['data'] == event['data']
                        )
                    ):
                        occured = True

                if occured is False:
                    meta_events.append(meta_event)

        meta_data['meta_events'] = meta_events

        return meta_data

    def _get_notes(track: mido.MidiTrack):
        """Calculates and returns the notes present in the input.

        Inspired by code found at in in pretty_midi/pretty_midi.py. Note
        that event.time in track must be in absolute units in order for
        this function to work correctly.

        Args:
            track (mido.MidiTrack): track with event.time in absolute time.

        Returns:
            list(tuple): list of notes as (note, start_tick, end_tick).
        """
        notes = []
        last_note_on = collections.defaultdict(list)

        for event in track:
            if event.is_meta is True:
                continue
            elif event.type == "note_on" and event.velocity > 0:
                last_note_on[event.note].append(event.time)
            elif event.type == "note_off" or (event.type == "note_on" and
                                              event.velocity == 0):
                # Ignore non-existent note-ons
                if event.note in last_note_on:
                    end_tick = event.time
                    open_notes = last_note_on[event.note]

                    notes_to_close = [
                        start_tick
                        for start_tick in open_notes
                        if start_tick != end_tick]
                    notes_to_keep = [
                        start_tick
                        for start_tick in open_notes
                        if start_tick == end_tick]

                    for start_tick in notes_to_close:
                        notes.append((event.note, start_tick, end_tick))

                    if len(notes_to_close) > 0 and len(notes_to_keep) > 0:
                        # Note-on on the same tick but we already closed
                        # some previous notes -> it will continue, keep it.
                        last_note_on[event.note] = notes_to_keep
                    else:
                        # Remove the last note on for this instrument
                        del last_note_on[event.note]

        return notes

    ticks_per_step = int(mid.ticks_per_beat/(div))

    # Convert event_time values in mid to absolute
    for track in mid.tracks:
        curr_tick = 0
        for event in track:
            event.time += curr_tick
            curr_tick = event.time

    # Get meta_data
    meta_data = _get_metadata(mid, ticks_per_step, div)

    # Get notes for all tracks
    mid_notes = []
    for track in mid.tracks:
        mid_notes += _get_notes(track)

    # Compute piano_roll
    piano_roll = collections.defaultdict(list)
    for note in mid_notes:
        for i in range(math.ceil(note[1]/ticks_per_step),
                       math.ceil(note[2]/ticks_per_step)):

            piano_roll[i].append(note[0])

    # Reformat
    piano_roll = [piano_roll.get(i, []) for i in range(max(piano_roll.keys()))]

    return PianoRoll(piano_roll, meta_data)


def from_pianoroll(piano_roll: PianoRoll):
    """Parses a PianoRoll object into a mid.MidiFile object.

    Args:
        piano_roll (PianoRoll): piano-roll to be parsed:

    Returns:
        mido.MidiFile: parsed MidiFile.
    """

    def _turn_on(track: mido.MidiTrack, notes: list):
        for note in notes:
            track.append(mido.Message('note_on', channel=0,
                         note=note, velocity=100, time=0))
            on_notes.append(note)

    def _turn_off(track: mido.MidiTrack, notes: list):
        for note in notes:
            track.append(mido.Message('note_off', channel=0,
                         note=note, velocity=100, time=0))
            on_notes.remove(note)

    mid = mido.MidiFile(type=0)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message('program_change', program=0, time=0))
    track.append(mido.MetaMessage('set_tempo', tempo=125000, time=0))

    tick = 240
    delta_t = 0
    on_notes = []
    for curr_notes in piano_roll:
        turn_on_notes = [note for note in curr_notes if note not in on_notes]
        turn_off_notes = [note for note in on_notes if note not in curr_notes]

        if turn_off_notes == [] and turn_on_notes == []:
            delta_t += tick
        else:
            ind = len(track)
            _turn_off(track, turn_off_notes)
            _turn_on(track, turn_on_notes)
            track[ind].time += delta_t
            delta_t = tick

    track.append(mido.MetaMessage('end_of_track', time=0))

    return mid
