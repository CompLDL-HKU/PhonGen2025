# Configuration parameters
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-4
READ_BASE_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025'
MODEL_LOAD_BASE_PATH = '/mnt/storage/franklhtan/projects/PhonGen2025'
WRITE_BASE_PATH = '/mnt/storage/franklhtan/projects/PhonGen2025'
NPY_BASE_PATH = '/mnt/storage/ldl_linguistics'
CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/metadata_train_phase1.csv'
CSV_PATH2 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase2_b/metadata_train_phase2_b.csv'
CSV_PATH3 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/metadata_test_b.csv'
MAX_SAMPLES = 50000000
DEVICE = 'cuda'
OUT_FEATURES = 4
TEMPERATURE = 0.07
MINMAX = False
# RUN_NAME = 'a_BS32_LR1E4_FEAT4_TEMP007V2_EU'
RUN_TIMES_START = 1
RUN_TIMES_END = 10
RUN_NAMES = ['b_BS32_LR1E4_FEAT4_TEMP007V2_EU_1AT50_both',
             'b_BS32_LR1E4_FEAT4_TEMP007V2_EU_1phaseONLY',]
            #  'b_BS32_LR1E4_FEAT4_TEMP007V2_EU_2AT50_both']