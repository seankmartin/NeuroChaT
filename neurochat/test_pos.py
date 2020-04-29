import numpy as np
import logging

file_name = r"D:\SubRet_recordings_imaging\muscimol_data\CanCSR8_muscimol\05102018\s4_big sq\05102018_CanCSR8_bigsq_10_4.pos"


def untracked(pos):
    return ((pos[0] == 1023) and (pos[1] == 1023))


def run(file_name):
    f = open(file_name, 'rb')
    total_pos_samples = 0
    while True:
        line = f.readline()
        try:
            line = line.decode('latin-1')
        except:
            break
        if line == '':
            break
        elif line.startswith("pos_format"):
            info = line.split(" ")[-1]
            if info[:-2] != "t,x1,y1,x2,y2,numpix1,numpix2":
                logging.error(
                    ".pos reading only supports 2-spot mode currently")
                print(info[:-2])
                print("t,x1,y1,x2,y2,numpix1,numpix2")
                return
        elif line.startswith("num_pos_samples"):
            total_pos_samples = int(line.strip().split(" ")[-1])
        elif line.startswith("timebase"):
            timebase = int(line.split(" ")[-2])
        elif line.startswith("pixels_per_metre"):
            pixels_per_metre = float(line.split(" ")[-1])
        elif line.startswith("data_start"):
            break

    f.seek(0, 0)
    header_offset = []
    while True:
        try:
            buff = f.read(10).decode('UTF-8')
        except:
            break
        if buff == 'data_start':
            header_offset = f.tell()
            break
        else:
            f.seek(-9, 1)
    f.seek(header_offset, 0)

    # save every 10 th of s_rate
    s_rate = timebase // 10
    rows = total_pos_samples
    big_pos = np.zeros(shape=(rows, 2), dtype=float)
    small_pos = np.zeros(shape=(rows, 2), dtype=float)
    spatial_data = np.zeros(shape=(rows, 5), dtype=float)
    with open("spatial.txt", "w") as wf:
        out_str = "time x/cm y/cm dir speed\n"
        wf.write(out_str)
        for sample_idx in range(total_pos_samples):
            chunk = f.read(20)
            frame_count = (
                16777216 * chunk[0] +
                65536 * chunk[1] +
                256 * chunk[2] +
                chunk[3])
            if frame_count != sample_idx:
                raise ValueError(
                    "Frame count {} does not match count {}".format(
                        frame_count, sample_idx))

            # Decode the words in the file
            words = np.zeros(shape=(7, ), dtype=float)
            for i in range(4, 18, 2):
                s_idx = (i - 4) // 2
                words[s_idx] = (256 * chunk[i]) + chunk[i + 1]

            # Words are:
            # big_spotx, big_spoty, little_spotx,
            # little_spoty, number_of_pixels_in_big_spot,
            # number_of_pixels_in_little_spot, total_tracked_pixels

            big_pos[sample_idx] = np.array(words[0:2])
            small_pos[sample_idx] = np.array(words[2:4])

        # Convert words to info
        # Interpolate untracked samples
        # TODO address going out of range
        for sample_idx in range(total_pos_samples):
            for pos_list in big_pos, small_pos:
                pos = pos_list[sample_idx]
                if untracked(pos):
                    ctr = 1
                    while untracked(pos_list[sample_idx + ctr]):
                        ctr += 1
                    last_pos = pos_list[sample_idx - 1]
                    next_tracked = pos_list[sample_idx + ctr]
                    difference = (next_tracked - last_pos)
                    interp = last_pos + (difference / (ctr + 1))
                    pos_list[sample_idx] = interp

        # Convert pixels to centimeters
        conversion = 100 / pixels_per_metre
        big_pos = big_pos * conversion
        small_pos = small_pos * conversion

        # Convert from raw positions
        time = 1 / timebase
        spatial_data[:, 0] = np.arange(0, total_pos_samples / timebase, time)
        x_pos = ((big_pos[:, 0] + small_pos[:, 0]) / 2)
        spatial_data[:, 1] = x_pos
        y_pos = ((big_pos[:, 1] + small_pos[:, 1]) / 2)
        spatial_data[:, 2] = y_pos
        distance_x = np.diff(x_pos)
        distance_y = np.diff(y_pos)
        from neurochat.nc_utils import angle_between_points
        for counter in range(total_pos_samples):
            angle = angle_between_points(
                small_pos[counter] + np.array([1, 0]),
                small_pos[counter], big_pos[counter])
            if big_pos[counter][1] > small_pos[counter][1]:
                angle = 360 - angle
            spatial_data[counter][3] = angle
        distance = np.sqrt(
            np.square(distance_x) + np.square(distance_y))
        speed = distance / time
        spatial_data[:, 4] = np.append(speed, 0.0)

        # Write out the information
        for counter in range(total_pos_samples):
            if counter % s_rate == 0:
                words = ""
                for val in spatial_data[counter]:
                    words = "{} {:.2f}".format(words, val)
                words = words[1:]
                wf.write(words + "\n")


run(file_name)
