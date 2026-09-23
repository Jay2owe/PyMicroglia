"""Recorded lifecycle events in the established review wording."""
EVENT_LABELS: dict[str, str] = {'present_at_start': 'already there when recording began', 'entered_field': 'entered through the edge of the field', 'born': "appeared as one of two, parent's area conserved", 'appeared': 'appeared mid-field, unexplained', 'present_at_end': 'still there when recording ended', 'left_field': 'left through the edge of the field', 'died': 'ground went dark, nothing left on it', 'absorbed': 'ground taken over by another cell', 'lost': 'name stopped, foreground stayed', 'vanished': 'name stopped, nothing to tell death from loss', 'divided': 'a second name separated from this one'}

def event_label(event):
    return EVENT_LABELS.get(event, str(event).replace("_", " "))
