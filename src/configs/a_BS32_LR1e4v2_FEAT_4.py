# Configuration parameters
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-4
CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/metadata_train_phase1.csv'
CSV_PATH2 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase2_a/metadata_train_phase2_a.csv'
CSV_PATH3 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/metadata_test_a.csv'
DEVICE = 'cuda'
OUT_FEATURES = 4
TEMPERATURE = 0.07
RUN_NAME = 'a_BS32_LR1E4_FEAT4_TEMP007V2.2'
SAMPLE_LIST = ['/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/aca/aca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/asa/asa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atca/atca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atsa/atsa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/acha/acha_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/acha/acha_0002.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/acha/acha_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/asha/asha_0001.npy', 
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/atcha/atcha_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/atsha/atsha_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/acha/acha_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/atcha/atcha_0001.npy', 
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/atsha/atsha_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/asha/asha_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/aca/aca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atca/atca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atsa/atsa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/asa/asa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/aca/aca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atsa/atsa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atsa/atsa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/acha/acha_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atca/atca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/atsha/atsha_0001.npy',   
               ]

