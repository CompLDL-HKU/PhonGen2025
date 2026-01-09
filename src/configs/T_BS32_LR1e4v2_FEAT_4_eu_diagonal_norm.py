# Configuration parameters
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-4
CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/metadata_train.csv'
# CSV_PATH2 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase2_a/metadata_train_phase2_a.csv'
CSV_PATH3 = '/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/metadata_test.csv'
DEVICE = 'cuda'
OUT_FEATURES = 4
TEMPERATURE = 0.07
SIMILARITY = "euclidean"
RUN_NAME = 'NORM_BS32_LR1E4_FEAT4_TEMP007V2_EU_DIAGONAL'
RUN_TIMES_START = 5
RUN_TIMES_END = 11
SAMPLE_LIST = [
    # =========================
    # s – c
    # =========================
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/isi/isi_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/ici/ici_0001.npy",

    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DsD/DsD_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DcD/DcD_0001.npy",

    # =========================
    # ts – tc
    # =========================
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/itsi/itsi_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/itci/itci_0001.npy",

    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DtsD/DtsD_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DtcD/DtcD_0001.npy",

    # =========================
    # s – tc
    # =========================
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/isi/isi_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/itci/itci_0001.npy",

    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DsD/DsD_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DtcD/DtcD_0001.npy",

    # =========================
    # ts – c
    # =========================
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/itsi/itsi_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/ici/ici_0001.npy",

    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DtsD/DtsD_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DcD/DcD_0001.npy",

    # =========================
    # ts – s
    # =========================
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/itsi/itsi_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/isi/isi_0001.npy",

    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DtsD/DtsD_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DsD/DsD_0001.npy",

    # =========================
    # tc – c
    # =========================
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/itci/itci_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/ici/ici_0001.npy",

    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DtcD/DtcD_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DcD/DcD_0001.npy",

    # =========================
    # Same-category pairs
    # =========================

    # s–s
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/isi/isi_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/isi/isi_0002.npy",

    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DsD/DsD_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DsD/DsD_0002.npy",

    # c–c
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/ici/ici_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/ici/ici_0002.npy",

    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DcD/DcD_0001.npy",
    "/mnt/storage/ldl_linguistics/PhonGen2025Diagonal/data/DcD/DcD_0002.npy",
]

