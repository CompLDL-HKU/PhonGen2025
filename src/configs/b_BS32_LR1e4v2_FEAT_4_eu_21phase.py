# Configuration parameters
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-4
CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/metadata_train_phase1.csv'
CSV_PATH2 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase2_b/metadata_train_phase2_b.csv'
CSV_PATH3 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/metadata_test_b.csv'
DEVICE = 'cuda'
OUT_FEATURES = 4
TEMPERATURE = 0.07
SIMILARITY = "euclidean"
RUN_NAME = 'b_BS32_LR1E4_FEAT4_TEMP007V2_EU_2AT50_both'
SAMPLE_LIST = ['/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/aca/aca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/asa/asa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atca/atca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atsa/atsa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/adja/adja_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/adja/adja_0002.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/adja/adja_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/adza/adza_0001.npy', 
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/aja/aja_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/aza/aza_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/adja/adja_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/aja/aja_0001.npy', 
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/aza/aza_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/adza/adza_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/aca/aca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atca/atca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atsa/atsa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/asa/asa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/aca/aca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atsa/atsa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atsa/atsa_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/adja/adja_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/atca/atca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_b/aza/aza_0001.npy',   
               ]

