# Configuration parameters
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-4
CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025_diag_new/data/data_train_unequal/metadata_train_unequal.csv'
# CSV_PATH2 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase2_a/metadata_train_phase2_a.csv'
CSV_PATH3 = '/mnt/storage/ldl_linguistics/PhonGen2025_diag_new/data/data_test_unequal/metadata_test_unequal.csv'
DEVICE = 'cuda'
OUT_FEATURES = 4
TEMPERATURE = 0.07
SIMILARITY = "euclidean"
RUN_NAME = '01_UNEQUAL_1E4_HID4_DIAGONAL_NORM'
RUN_TIMES_START = 1
RUN_TIMES_END = 11
SAMPLE_LIST = []
