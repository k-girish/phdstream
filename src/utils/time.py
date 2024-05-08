import time
import uuid

time_key_map = {}


def start_time_record(time_key=None):
    uuid_key = str(uuid.uuid1())
    if time_key is None:
        time_key = uuid_key
    else:
        time_key = f"{time_key}_{uuid_key}"
    if time_key in time_key_map:
        raise Exception(f"Time key {time_key} already exists")

    # Start time recording
    time_key_map[time_key] = {
        'start': time.time(),
        'lap': time.time()
    }

    return time_key


def end_time_record(time_key, print_duration=False, extra_msg=None):
    if time_key not in time_key_map:
        raise Exception(f"Time key {time_key} does not exists")

    # End time recording
    time_key_map[time_key]['end'] = time.time()
    duration = time_key_map[time_key]['end'] - time_key_map[time_key]['lap']
    time_key_map[time_key]['duration'] = duration
    if print_duration:
        if extra_msg is not None:
            print(extra_msg)
        print(
            f"Duration for {time_key} = {pretty_print_duration(duration)}"
        )

    time_key_map[time_key]['lap'] = time_key_map[time_key]['end']
    return duration


def pretty_print_duration(duration):
    return time.strftime("%H:%M:%S", time.gmtime(duration))
