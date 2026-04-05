# Configuration parameters
# Story: 02 is using large variance (std = 15), not using normalization, but keeping positions the same as 01. 
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-4
DATA_COLLECTION_NAME = 'data_LV'
CSV_PATH = f'/mnt/storage/ldl_linguistics/PhonGen2025_diag_new/{DATA_COLLECTION_NAME}/data_train_equal/metadata_train_equal.csv'
# CSV_PATH2 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase2_a/metadata_train_phase2_a.csv'
CSV_PATH3 = f'/mnt/storage/ldl_linguistics/PhonGen2025_diag_new/{DATA_COLLECTION_NAME}/data_test_equal/metadata_test_equal.csv'
DEVICE = 'cuda'
OUT_FEATURES = 4
TEMPERATURE = 0.07
SIMILARITY = "euclidean"
RUN_NAME = '02_EQUAL_1E4_HID4_DIAGONAL_NORM'
RUN_TIMES_START = 1
RUN_TIMES_END = 11
SAMPLE_LIST = []
NORMALIZE = False
