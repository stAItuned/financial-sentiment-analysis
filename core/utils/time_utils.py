import time


def spent_time(start_time, end_time):
    minutes = (end_time - start_time) // 60
    seconds = (end_time - start_time) - (minutes * 60)

    return ' {:.0f} min {:.0f} sec'.format(minutes, seconds)


def timestamp():
    time_stamp = f'{time.time():.0f}'

    return time_stamp
