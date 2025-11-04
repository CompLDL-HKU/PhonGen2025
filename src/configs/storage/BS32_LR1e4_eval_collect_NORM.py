# Configuration parameters
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
MAX_SAMPLES = 50000000
CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025/data/metadata.csv'
NPY_BASE_PATH = '/mnt/storage/ldl_linguistics'
READ_BASE_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025'
WRITE_BASE_PATH = '/mnt/storage/franklhtan/projects/PhonGen2025'
DEVICE = 'cuda'
OUT_FEATURES = 4
TEMPERATURE = 0.07
MINMAX = True
RUN_NAME = 'BS32_LR1E4_FEAT4_TEMP007V2_NORM'
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