import argparse
import sys
import pandas as pd
from pipelines import BankImprovedPipeline
from pipelines import BankVanillaPipeline
from pipelines import GermanPipeline
from pipelines import ImprovedGermanPipeline


def get_args(args=None):
    parser = argparse.ArgumentParser(description='ML Pipeline')
    parser.add_argument('-p', '--path',
                        help='path to dataset',
                        required='True')
    parser.add_argument('-br', '--balance_ratio',
                        help='balance threshold',
                        default=0.7, type=float)
    parser.add_argument('-c', '--client',
                        help='client name',
                        required=True, type=str)
    parser.add_argument('-v', '--vanilla',
                        help='run vanilla pipeline in addition to improvement pipeline',
                        default=False, type=bool)
    return parser.parse_args(args)


def load_bank_marketing_data(path):
    return pd.read_csv(path, delimiter=";", header='infer')


def load_german_risk_data(path):
    names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
             'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
             'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
             'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']
    return pd.read_csv(path, names=names, delimiter=' ')


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    if args.client == 'bank':
        data = load_bank_marketing_data(args.path)
        if args.vanilla:
            pipe_vanilla = BankVanillaPipeline("Bank Vanilla", data)
            pipe_vanilla.run()
        pipe = BankImprovedPipeline("Bank Improved Pipeline", data, seed=2, balance_ratio=args.balance_ratio)
        pipe.run()
    if args.client == 'german':
        data = load_german_risk_data(args.path)
        if args.vanilla:
            pipe_vanilla = GermanPipeline("German Vanilla", data)
            pipe_vanilla.run()
        pipe = ImprovedGermanPipeline("German Improved Pipeline", data, seed=2, balance_ratio=args.balance_ratio)
        pipe.run()
