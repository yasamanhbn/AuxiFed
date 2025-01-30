class Config():

    def __init__(self, DATASETTYPE, alpha, local_iter=4, rounds=4, gan_epochs=50, adverseial_training=False, adverseial_training_probs=2, replace_probs=16):
        self.latent_dim = 100
        self.img_size=28
        self.channels=1
        self.img_shape = (self.channels, self.img_size, self.img_size)

        self.DATASETTYPE = DATASETTYPE  #'EMnist', 'Mnist'
        self.ALPHA = alpha # 0.2, 1
        self.lr = 0.001
        self.NUM_CLIENTS = 8
        self.LOCAL_ITERS = local_iter
        self.NUM_EPOCHS = gan_epochs
        self.gan_epoch = gan_epochs  
        self.BATCH_SIZE = 64
        self.adverseial_training = adverseial_training  #Adverserial Example Training

        self.adverseial_training_probs = adverseial_training_probs
        self.replace_probs = replace_probs

        if self.DATASETTYPE =='Mnist' and self.ALPHA==0.2:
            self.train_len = [7709, 8908, 8482, 9768, 4541, 6813, 3468, 10311] #Mnist alpha0.2
            self.test_len = [368, 997, 757, 390, 1245, 1116, 877, 1119]
            self.datasetPath = "Mnist-u8c10-alpha0.2-ratio1"
            self.CLASS_NUM = 10

            self.Results = "./mnist-alpha0.2-pgd-vaeReport.csv"
            self.save_checkpoint = "./mnist-alpha0.2-vaeWeight-pgd"

        elif self.DATASETTYPE =='Mnist' and self.ALPHA==1:
            self.train_len = [8217, 7619, 6034, 9426, 7497, 3937, 6952, 10318] #MNISt-alpha1
            self.test_len = [878, 878, 1245, 1119, 1245, 1119, 1245, 1245]
            self.datasetPath = "Mnist-u8c10-alpha1-ratio1"
            self.CLASS_NUM = 10

            self.Results = "./mnist-alpha1-pgd-ACganReport.csv"
            self.save_checkpoint = "./mnist-pgd-alpha1-ACGanWeight"

        elif self.DATASETTYPE =='EMnist' and self.ALPHA==0.2:
            self.train_len = [16476, 13422, 15823, 15763, 17408, 16733, 16445, 12730]
            self.test_len = [1600, 1900, 1500, 1600, 1900, 1900, 1600, 2600]
            self.datasetPath = "EMnist-u8c27-alpha0.2-ratio1"
            self.CLASS_NUM = 27

            self.Results = "./emnist-alpha0.2-cganReport.csv"
            self.save_checkpoint = "./emnist-alpha0.2-modelCGanWeight"

        elif self.DATASETTYPE =='EMnist' and self.ALPHA==1:
            self.train_len = [15784, 15629, 11810, 17649, 16154, 15673, 14891, 17210] #EMNIST alpha1
            self.test_len = [2200, 2600, 2600, 2400, 2500, 2300, 2600, 2600] #EMNIST alpha1
            self.datasetPath = "EMnist-u8c27-alpha1-ratio1"
            self.CLASS_NUM = 27

            self.Results = "./emnist-alpha1-pgd-vaeReport.csv"
            self.save_checkpoint = "./emnist-alpha1-vaeModelWeight-pgd"
