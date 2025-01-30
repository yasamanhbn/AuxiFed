class USER():
    def __init__(self, id, train_data, test_data):
        self.id = id
        self.train_data=train_data
        self.test_data=test_data
        self.class_dict = {}

    def clean_data(self):
        for t_d in self.train_data:
            class_label = t_d[1].item()

            # Check if the class label exists in the dictionary
            if class_label in self.class_dict:
                self.class_dict[class_label].append(t_d[0])
            else:
                self.class_dict[class_label] = [t_d[0]]