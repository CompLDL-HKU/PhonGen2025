# Configuration parameters
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-4
READ_BASE_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025'
WRITE_BASE_PATH = '/mnt/storage/franklhtan/projects/PhonGen2025'
NPY_BASE_PATH = '/mnt/storage/ldl_linguistics'
CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/metadata_train_phase1.csv'
CSV_PATH2 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase2_a/metadata_train_phase2_a.csv'
CSV_PATH3 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/metadata_test_a.csv'
MAX_SAMPLES = 50000000
DEVICE = 'cuda'
OUT_FEATURES = 4
TEMPERATURE = 0.07
MINMAX = False
# RUN_NAME = 'a_BS32_LR1E4_FEAT4_TEMP007V2_EU'
RUN_NAMES = ['a_BS32_LR1E4_FEAT4_TEMP007V2_EU_1AT50_both',
             'a_BS32_LR1E4_FEAT4_TEMP007V2_EU_1phaseONLY',
             'a_BS32_LR1E4_FEAT4_TEMP007V2_EU_2AT50_both']