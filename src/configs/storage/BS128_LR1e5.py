# Configuration parameters
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-5
MAX_SAMPLES = 50000000
CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025/data/metadata.csv'
NPY_BASE_PATH = '/mnt/storage/ldl_linguistics'
DEVICE = 'cuda'
OUT_FEATURES = 128
TEMPERATURE = 0.07
RUN_NAME = 'BS128_LR1E5_FEAT128_TEMP007'
SAMPLE_LIST = ['/mnt/storage/ldl_linguistics/PhonGen2025/data/itsi/itsi_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/itsi/itsi_0002.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/itsi/itsi_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/etse/etse_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/itsi/itsi_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/ici/ici_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/itsi/itsi_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/ece/ece_0001.npy', 
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LtsL/LtsL_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LtsL/LtsL_0002.npy', 
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LtsL/LtsL_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/FtsF/FtsF_0001.npy', 
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LtsL/LtsL_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LcL/LcL_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LtsL/LtsL_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/FcF/FcF_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LtcL/LtcL_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LtcL/LtcL_0002.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LtcL/LtcL_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/FtcF/FtcF_0001.npy', 
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LtcL/LtcL_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LsL/LsL_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/LsL/LsL_0001.npy',
               '/mnt/storage/ldl_linguistics/PhonGen2025/data/FtcF/FtcF_0001.npy',   
               ]