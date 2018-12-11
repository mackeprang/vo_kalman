def get_im_and_data_dir(mission_num):
    if mission_num == 1:
        # Mission 1

        imdir = 'Master Thesis/Pictures/20181010_110156.2920_Mission_1'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_110156_Mission_1/output.h5'
    elif mission_num == 2:
#        Mission 2
        imdir = '/Master Thesis/Pictures/20181010_115518.5820_Mission_2'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_115518_Mission_2/output.h5'
    elif mission_num == 3:
        # Mission 3
        imdir = '/Master Thesis/Pictures/20181010_111238.6910_Mission_3'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_111238_Mission_3/output.h5'
    elif mission_num == 4:
        # Mission 4
        imdir = 'Master Thesis/Pictures/20181010_111618.6170_Mission_4'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_111618_Mission_4/output.h5'
    elif mission_num == 5:
        # Mission 5
        imdir = 'Master Thesis/Pictures/20181010_112029.5110_Mission_5'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_112029_Mission_5/output.h5'
    elif mission_num == 6:
        # Mission 6
        imdir = '/Master Thesis/Pictures/20181010_123123.4130_Mission_6'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_123123_Mission_6/output.h5'
    elif mission_num == 7:
        # Mission 7
        imdir = '/Master Thesis/Pictures/20181010_124010.1250_Mission_7'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_124010_Mission_7/output.h5'
    elif mission_num == 8:
        # Mission 8
        imdir = '/Master Thesis/Pictures/20181010_124358.5230_Mission_8'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_124358_Mission_8/output.h5'
    elif mission_num == 9:
        # Mission 9
        imdir = 'Master Thesis/Pictures/20181010_125259.7070_Mission_9'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_125259_Mission_9/output.h5'
    elif mission_num == 10:
        # Mission 10
        imdir = '/Master Thesis/Pictures/20181010_125401.0250_Mission_10'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_125401_Mission_10/output.h5'
    elif mission_num == 11:
        # Mission 11
        imdir = '/Master Thesis/Pictures/20181010_125949.2330_Mission_11'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_125949_Mission_11/output.h5'
    elif mission_num == 12:
        # Mission 12
        imdir = '/Master Thesis/Pictures/20181010_112638.9150_Mission_6'
        datadir = 'Master Thesis/Data/20180910 Optical flowtest/20181010_112638_Mission_6/output.h5'
    elif mission_num == 13:
        # New collection of data - mission 4:
        imdir = 'Master Thesis/Test Pictures/20181105_120815.7780_Mission_4'
        datadir = 'Master Thesis/Test Data/20181105_120815_Mission_4/output.H5'

    elif mission_num == 14:
        # New collection of data - mission 4:
        imdir = 'Master Thesis/Test Pictures/20181105_120412.4490_Mission_3'
        datadir = 'Master Thesis/Test Data/20181105_120412_Mission_3/output.H5'

    elif mission_num == 15:
        # New collection of data - mission 4:
        imdir = 'data'
        datadir = 'data/output.H5'
    else:
        imdir = []
        datadir = []

    return imdir,datadir
