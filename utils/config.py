class paths:
    split_dir = "dataset/split_info"
    extract_dir = "dataset/extract"
    work_dir = "dataset/work"

    smpl_m = "smpl_model/SMPL_male.pkl"

    raw_dipimu_dir = "dataset\DIP_IMU"
    dipimu_dir = "data"

    raw_amass_dir = "dataset\AMASS"
    amass_dir = "data/dataset_work/AMASS"


class sensor6Info:
    model_path = "model_log/best_model6.pth"
    sensor_idx = [18, 19, 4, 5, 15, 0]
    reduced_idx = [1, 2, 3, 6, 9, 12, 13, 14, 16, 17]
    input_idx = [0, 1, 2, 3, 4, 5]


class sensor5Info:
    model_path = "model_log/best_model5.pth"
    sensor_idx = [18, 19, 4, 5, 15]
    reduced_idx = [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17]
    input_idx = [0, 1, 2, 3, 4]


class sensor5Infohpwwl:
    model_path = "model_log/best_model5_hpwwl.pth"
    sensor_idx = [18, 19, 4,  15,0 ]
    reduced_idx = [1, 2, 3, 5, 6, 9, 12, 13, 14, 16, 17]
    input_idx = [0, 1, 2, 4, 5]

class sensor4Info:
    model_path = "model_log/best_model4.pth"
    sensor_idx = [18, 19, 4, 5]
    reduced_idx = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17]
    input_idx = [0, 1, 2, 3]


class sensor3Info:
    model_path = "model_log/best_model3.pth"
    sensor_idx = [18, 19, 15]
    reduced_idx = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17]
    input_idx = [0, 1, 4]


class sensor2Info:
    model_path = "model_log/best_model2.pth"
    sensor_idx = [18, 19]
    reduced_idx = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17]
    input_idx = [0, 1]
