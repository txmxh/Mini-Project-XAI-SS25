
print("Number of classes:", num_classes)
print("Train label counts:", torch.bincount(data.train_y))
print("Test label counts :", torch.bincount(data.test_y))
