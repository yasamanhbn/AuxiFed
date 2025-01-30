DATA_DIR = '/'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FLATTEN_SIZE = 16*7*7