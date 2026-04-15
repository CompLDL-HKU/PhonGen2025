# Configuration parameters
# Story: 03 is using original variance (std = 5), not using normalization, but changing COG to (0, 40) while FD being still (30, 70). 
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-4
DATA_COLLECTION_NAME = 'data_SC'
CSV_PATH = f'/mnt/storage/ldl_linguistics/PhonGen2025_diag_new/{DATA_COLLECTION_NAME}/data_train_equal/metadata_train_equal.csv'
# CSV_PATH2 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase2_a/metadata_train_phase2_a.csv'
CSV_PATH3 = f'/mnt/storage/ldl_linguistics/PhonGen2025_diag_new/{DATA_COLLECTION_NAME}/data_test_equal/metadata_test_equal.csv'
DEVICE = 'cuda'
OUT_FEATURES = 4
TEMPERATURE = 0.07
SIMILARITY = "euclidean"
RUN_NAME = '03_EQUAL_1E4_HID4_DIAGONAL_NORM'
RUN_TIMES_START = 1
RUN_TIMES_END = 11
SAMPLE_LIST = []
NORMALIZE = False
