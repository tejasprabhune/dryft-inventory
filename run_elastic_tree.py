from torch.utils.data import DataLoader
from dryft.models import ElasticDecisionTree
from dryft.data.part_dataset import DTPartDataset

model = ElasticDecisionTree(reset=False, verbose=True)

train_dataset = DTPartDataset('train_data_sample_no_mts.csv')
test_dataset = DTPartDataset('test_data_sample.csv')

train_dataloader = DataLoader(train_dataset, batch_size=1000)
train_data = next(iter(train_dataloader))

test_dataloader = DataLoader(test_dataset, batch_size=1000)
test_data = next(iter(test_dataloader))

model.fit(train_dataset)
model.save("dryft/ckpts/naive_dt.pkl")

model.fit_to_feedback(train_dataset)


train_preds = model(train_data[0])
test_preds = model(test_data[0])

print("DT Train accuracy: ", (train_preds == train_data[1]).float().mean().item())
print("DT Test accuracy: ", (test_preds == test_data[1]).float().mean().item())

print(train_preds.numpy().astype(int).tolist())
print(train_data[1].numpy().astype(int).tolist())

print(test_preds.numpy().tolist())
print(test_data[1].numpy().tolist())
