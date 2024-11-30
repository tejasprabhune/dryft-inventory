import argparse

from torch.utils.data import DataLoader
from dryft.models import ElasticDecisionTree
from dryft.data.part_dataset import DTPartDataset

parser = argparse.ArgumentParser(description='Train Elastic Decision Tree')
parser.add_argument('--reset', action='store_true', help='Reset the rules database')
parser.add_argument('--verbose', action='store_true', help='Print verbose output')
parser.add_argument('--rules_db_path', type=str, default='rules_db', help='Path to the rules database')
parser.add_argument('--ckpt', type=str, default=None, help='Path to the checkpoint')
parser.add_argument('--output_ckpt', type=str, default='output_ckpt', help='Path to the output checkpoint')
parser.add_argument('--train_data', type=str, default='train_data_sample.csv', help='Path to the training data')
parser.add_argument('--test_data', type=str, default='test_data_sample.csv', help='Path to the test data')

args = parser.parse_args()

e_model = ElasticDecisionTree(reset=args.reset, verbose=args.verbose, rules_db_path=args.rules_db_path, ckpt=args.ckpt)

train_dataset = DTPartDataset(args.train_data)
test_dataset = DTPartDataset(args.test_data)

train_dataloader = DataLoader(train_dataset, batch_size=1000)
train_data = next(iter(train_dataloader))

test_dataloader = DataLoader(test_dataset, batch_size=1000)
test_data = next(iter(test_dataloader))

e_model.fit(train_dataset)
e_model.save(args.output_ckpt)

e_model.fit_to_feedback(train_dataset)

train_preds = e_model(train_data[0])
test_preds = e_model(test_data[0])

print("EDT Train accuracy: ", (train_preds == train_data[1]).float().mean().item())
print("EDT Test accuracy: ", (test_preds == test_data[1]).float().mean().item())

print(train_preds.numpy().astype(int).tolist())
print(train_data[1].numpy().astype(int).tolist())

print(test_preds.numpy().tolist())
print(test_data[1].numpy().tolist())
